import argparse
from pathlib import Path
import pandas as pd
import json
import pickle
from math import floor
import logging
from datetime import datetime
import numpy as np

from gluonts.dataset.split import split
from gluonts.dataset.pandas import PandasDataset
from gluonts.torch import TemporalFusionTransformerEstimator
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import _to_dataframe

# FIXME: Record random seeds.

MODEL='tft'
TEST_LENGTH = int(0.2 * 200) 
VALIDATION_LENGTH = int(0.1 * 200)
EVAL_DURATION = {
    "val": VALIDATION_LENGTH,
    "test": TEST_LENGTH
}
QUANTILES = [0.005, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.975, 0.995]


def get_args():
    parser = argparse.ArgumentParser(
        description=f'Train and evaluate an {MODEL} forcaster.'
        )
    # Window-related arguments
    parser.add_argument('--prediction_length', type=int, default=3, help='Length of the prediction horizon.')
    parser.add_argument('--context_multiplier', type=int, default=3, help='The multiplication factor to determine the context window length.')
    parser.add_argument('--step_size', type=int, default=1, help='The difference in starting points of two consecutive sliding windows.')
    # DL Training-related arguments
    parser.add_argument('--hp_tuning', action='store_true', help='Determine whether the script is run for hyperparameter tuning or not.')
    parser.add_argument('--num_epochs', type=int, default=5, help='The maximum number of epochs to train the model.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate of the model during training.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training.')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value (max norm) during training.')
    # Model-related arguments
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--hidden_dim', type=int, default=40, help='Number of hidden states in the LSTM and attention layer.')
    parser.add_argument('--variable_dim', type=int, default=20, help='Number of feature embeddings.')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate.')
    # Data-related arguments
    parser.add_argument('--data_dir', type=Path, default=Path().cwd().joinpath('data'), help='The path to the folder that contains the data.')
    parser.add_argument('--output_dir', type=Path, default=Path().cwd().joinpath('output'), help='The path of the folder to record the output.')

    args = parser.parse_args()

    return args

def setup_logger(file_name: str, output_directory: Path = 'results', file_log_level='DEBUG', stream_log_level='INFO'):
    """Initilizes and formats the root logger. It also sets the log
    levels for the log file and stream handler.
    """
    # Create the results folder if it does not exist.
    output_directory.mkdir(parents=True, exist_ok=True)
    # Setup logger.
    logger = logging.getLogger(__name__)

    log_id = file_name + '.log'
    log_file = str(output_directory.joinpath(log_id))
    # log_file = os.path.join(output_directory, log_id)
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')

    # Set logger's logging level.
    if file_log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif file_log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif file_log_level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif file_log_level == 'ERROR':
        logger.setLevel(logging.ERROR)
    else:
        raise ValueError(
            "file_log_level can only be DEBUG, INFO, WARNING or ERROR.")

    # Initialize and format the stream_handler.
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    stream_handler.setFormatter(formatter)

    # Set stream_handler logging level.
    if stream_log_level == 'DEBUG':
        stream_handler.setLevel(logging.DEBUG)
    elif stream_log_level == 'INFO':
        stream_handler.setLevel(logging.INFO)
    elif stream_log_level == 'WARNING':
        stream_handler.setLevel(logging.WARNING)
    elif stream_log_level == 'ERROR':
        stream_handler.setLevel(logging.ERROR)
    else:
        raise ValueError(
            "stream_log_level can only be DEBUG, INFO, WARNING or ERROR.")

    logger.addHandler(stream_handler)

    return logger

def setup_output_folder(args, timestamp) -> Path:
    output_folder = args.output_dir.joinpath(f'{timestamp}_{MODEL}_lr_{args.learning_rate}_epochs_{args.num_epochs}_bsize_{args.batch_size}_gradclip_{args.clip_grad}_nchannels_{args.num_channels}_nmlplayer_{args.num_mlp_layers}_nneuron_{args.neuron_per_layer}')
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder

def create_dataset(
        data_dir:Path,
        hp_tuning:bool,
        ts_dict_name:str='act_dataset_dict_v2.pkl',
        static_features_name:str='static_features_v1.pkl'
    ):
    # Import dataset
    # Load data from pickled files into a PandasDataset
    with open(data_dir.joinpath(ts_dict_name), 'rb') as d:
        ts_dict = pickle.load(d)

    # Change the type of dynamic real values to float32.
    for df in ts_dict.values():
        df['safety_metric_cte_normalized_z_score'] = df['safety_metric_cte_normalized_z_score'].astype(np.float32)
        df['estimated_distance_to_centerline_meters_normalized_z_score'] = df['estimated_distance_to_centerline_meters_normalized_z_score'].astype(np.float32)
        df['estimated_heading_error_degrees_normalized_z_score'] = df['estimated_heading_error_degrees_normalized_z_score'].astype(np.float32)

    static_features = pd.read_pickle(data_dir.joinpath(static_features_name))

    if hp_tuning:
        #FIXME: cast the item_id in the saved dataset to int.
        dataset = PandasDataset(
            {int(item_id): df[:-1*EVAL_DURATION['test']] for item_id, df in ts_dict.items()},
            target='safety_metric_cte_normalized_z_score',
            past_feat_dynamic_real=[
                'estimated_distance_to_centerline_meters_normalized_z_score',
                'estimated_heading_error_degrees_normalized_z_score'
            ],
            static_features=static_features,
        )
    else:
        dataset = PandasDataset(
            {int(item_id): df for item_id, df in ts_dict.items()},
            target='safety_metric_cte_normalized_z_score',
            past_feat_dynamic_real=[
                'estimated_distance_to_centerline_meters_normalized_z_score',
                'estimated_heading_error_degrees_normalized_z_score'
            ],
            static_features=static_features,
        )
    return dataset

def count_windows(duration:int, window_size:int, step_size:int) -> int:
    """Counts the number of windows that can be generated over a duration.
    """
    return (duration - window_size) // step_size + 1

def save_output(output_dir:Path, model, agg_metrics, item_metrics, logger) -> None:
    # Save trained model
    model_path = output_dir.joinpath('model')
    model_path.mkdir(parents=True, exist_ok=True)
    model.serialize(model_path)
    logger.info('Model saved.')

    # Save metrics
    agg_metrics_json_obj = json.dumps(agg_metrics, indent=4)
    
    with open(output_dir.joinpath('aggregate_metrics.json'), "w") as outfile:
        outfile.write(agg_metrics_json_obj)
    
    item_metrics.to_pickle(output_dir.joinpath('item_metrics.pkl'))
    logger.info('Metrics recorded.')
    return

def main():
    args = get_args()
    # Get current timestamp to use as a unique ID.
    timestamp = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_folder = setup_output_folder(args, timestamp)
    logger = setup_logger(file_name=f'{timestamp}_{MODEL}_HP', output_directory=output_folder, file_log_level='DEBUG', stream_log_level='INFO')
    logger.info(f'Job started (timestamp={timestamp})')
    logger.debug(f'The passed arguments are: {vars(args)}')
    context_length = args.context_multiplier * args.prediction_length
    logger.debug(f'Context length is: {context_length}')

    if args.hp_tuning:
        mode = 'val'
        logger.info('Running hp tuning.')
    else:
        mode = 'test'

    dataset = create_dataset(args.data_dir, args.hp_tuning)
    logger.info('Dataset created.')

    logger.debug(f'Offset value for dataset split during {mode}:{-1*EVAL_DURATION[mode]}')
    # Split dataset
    training_data, test_template = split(dataset, offset=-1*EVAL_DURATION[mode])
    test_data = test_template.generate_instances(
        prediction_length=args.prediction_length,
        windows=count_windows(EVAL_DURATION[mode], args.prediction_length, args.step_size),
        distance=args.step_size,
        max_history=context_length
    )
    logger.info('Dataset split completed')

    # Instantiate and train the estimator
    model = TemporalFusionTransformerEstimator(
        freq='S',
        prediction_length=args.prediction_length,
        context_length=context_length,
        quantiles=QUANTILES,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        # variable_dim=args.variable_dim,
        static_dims=[1, 1],
        past_dynamic_dims=[1, 1],
        static_cardinalities=[3, 4],
        lr=args.learning_rate,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        trainer_kwargs={
            'epochs': args.num_epochs,
            'gradient_clip_val': args.clip_grad,
        },
    ).train(training_data)
    logger.info('Training completed.')

    # Generate predictions
    logger.info('Generating forecasts.')
    forecasts = list(model.predict(test_data.input))
    logger.info('Forecasts generated.')
    
    # Cast the test data into a list to be used by the evaluator.
    test_data_list = list(map(_to_dataframe, test_data))
    logger.debug(f'Number of test samples: {len(test_data_list)}')


    evaluator = Evaluator(quantiles=QUANTILES)
    agg_metrics, item_metrics = evaluator(test_data_list, forecasts)
    logger.info('Evaluation completed.')

    logger.info('Saving outputs.')
    save_output(output_folder, model, agg_metrics, item_metrics, logger)

    # Add the args and agg_metrics to a dataframe and save it.
    # args_df = pd.DataFrame({k: [v] for k, v in vars(args).items()})
    # agg_metric_df = pd.DataFrame({'model':MODEL, **agg_metrics}, index=[0])
    # output_df = pd.concat([args_df, agg_metric_df], axis=1)
    output_df = pd.DataFrame({'model':MODEL, **vars(args), **agg_metrics}, index=[0])
    output_df.to_pickle(output_folder.joinpath('output_df.pkl'))
    logger.info('Outputs saved.')

    logger.info('Job completed.')

if __name__=='__main__':
    main()
