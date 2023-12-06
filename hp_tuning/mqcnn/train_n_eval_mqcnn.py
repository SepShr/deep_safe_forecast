import argparse
from pathlib import Path
import pandas as pd
import json
import pickle
from math import floor
import logging
from datetime import datetime
import tracemalloc
import time

from gluonts.dataset.split import split
from gluonts.dataset.pandas import PandasDataset
from gluonts.mx import MQCNNEstimator, Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import _to_dataframe

# FIXME: Record random seeds.

MODEL='mqcnn'
TEST_LENGTH = int(0.2 * 200) 
VALIDATION_LENGTH = int(0.1 * 200)
EVAL_DURATION = {
    "val": VALIDATION_LENGTH,
    "test": TEST_LENGTH
}
QUANTILES = [0.005, 0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.975, 0.995]
NUM_CNN_LAYERS = 3


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
    parser.add_argument('--num_mlp_layers', type=int, default=1, help='Number of the MLP decoder layers.')
    parser.add_argument('--neuron_per_layer', type=int, default=30, help='Number of neurons per hidden layer in the MLP decoder.')
    parser.add_argument('--num_channels', type=int, default=30, help='Number of channels in each layer of the CNN encoder.')
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
        ts_dict_name:str='act_dataset_dict_v1.pkl',
        static_features_name:str='static_features_v1.pkl'
    ):
    # Import dataset
    # Load data from pickled files into a PandasDataset
    with open(data_dir.joinpath(ts_dict_name), 'rb') as d:
        ts_dict = pickle.load(d)

    static_features = pd.read_pickle(data_dir.joinpath(static_features_name))

    if hp_tuning:
        #FIXME: cast the item_id in the saved dataset to int.
        dataset = PandasDataset(
            {int(item_id): df[:-1*EVAL_DURATION['test']] for item_id, df in ts_dict.items()},
            target='safety_metric_cte',
            past_feat_dynamic_real=[
                'estimated_distance_to_centerline_meters',
                'estimated_heading_error_degrees'
            ],
            static_features=static_features,
        )
    else:
        dataset = PandasDataset(
            {int(item_id): df for item_id, df in ts_dict.items()},
            target='safety_metric_cte',
            past_feat_dynamic_real=[
                'estimated_distance_to_centerline_meters',
                'estimated_heading_error_degrees'
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
    logger = setup_logger(file_name=f'{timestamp}_MQCNN_HP', output_directory=output_folder, file_log_level='DEBUG', stream_log_level='INFO')
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
    model = MQCNNEstimator(
        freq='S',
        prediction_length=args.prediction_length,
        context_length=context_length,
        use_feat_static_cat=True,
        cardinality=[3, 4],
        use_past_feat_dynamic_real=True,
        channels_seq=[args.num_channels for _ in range(NUM_CNN_LAYERS)],
        decoder_mlp_dim_seq=[args.neuron_per_layer for _ in range(args.num_mlp_layers)],
        quantiles=QUANTILES,
        enable_decoder_dynamic_feature=False,
        # seed=,  # NOTE: Remember to set this for final experiments.
        trainer=Trainer(
            ctx="cpu",
            epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            clip_gradient=args.clip_grad,
            # weight_decay=WEIGHT_DECAY,
            hybridize=False,
            num_batches_per_epoch=args.batch_size
        )
    ).train(training_data)
    logger.info('Training completed.')

    # Generate predictions
    logger.info('Generating forecasts.')
    tracemalloc.start()
    start_time = time.perf_counter_ns()
    forecasts = list(model.predict(test_data.input))
    elapsed_time = time.perf_counter_ns() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    avg_inference_time = elapsed_time / len(forecasts)
    peak_memory_mb = peak_memory / (1024 * 1024)
    perf_metrics = {
        'elapsed_time_ns': elapsed_time,
        'avg_inference_time_ns': avg_inference_time,
        'peak_memory': peak_memory,
        'peak_memory_mb': peak_memory_mb
    }
    logger.info('Forecasts generated.')
    logger.info(f'Performance metrics: {perf_metrics}')
    
    # Cast the test data into a list to be used by the evaluator.
    test_data_list = list(map(_to_dataframe, test_data))
    logger.debug(f'Number of test samples: {len(test_data_list)}')


    evaluator = Evaluator(quantiles=QUANTILES)
    agg_metrics, item_metrics = evaluator(test_data_list, forecasts)
    logger.info('Evaluation completed.')

    logger.info('Saving outputs.')
    save_output(output_folder, model, agg_metrics, item_metrics, logger)

    # Save forecasts to a pickle file.
    with open(output_folder.joinpath('forecasts_list.pkl'), 'rb') as flf:
        pickle.dump(forecasts, flf)
    logger.info('Forecasts saved.')
    
    # Save test_data_list to a pickle file.
    with open(output_folder.joinpath('test_data_list.pkl'), 'wb') as tlf:
        pickle.dump(test_data_list, tlf)
    logger.info('Test data saved.')

    # Add the args and agg_metrics to a dataframe and save it.
    args_df = pd.DataFrame({k: [v] for k, v in vars(args).items()})
    agg_metric_df = pd.DataFrame({'model':MODEL, **agg_metrics, **perf_metrics}, index=[0])
    output_df = pd.concat([args_df, agg_metric_df], axis=1)
    output_df.to_pickle(output_folder.joinpath('output_df.pkl'))
    logger.info('Outputs saved.')

    logger.info('Job completed.')

if __name__=='__main__':
    main()
