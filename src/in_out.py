import json
import logging

import joblib
import pandas as pd

from src import directories
from src.constants import filenames
from src.constants.columns import COLUMNS
from src.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)

def get_data(config):

    logger.info("Getting dataset.")

    ## Download input file from BCG Egnyte - RAI folder
    ## https://bcg01.egnyte.com/navigate/folder/881ba5c2-8f6e-4eb6-b940-848faa78dee9
    input_path = directories.raw_data_dir / config.input_data,
    dataset = load_dataset(input_path[0], config)
    logger.info(f"Dataset loaded with {len(dataset)} rows.")

    return dataset

def save_dataset(dataset, *, path):
    dataset.to_csv(path, index=False)
    logger.info(f"Dataset saved at {path.relative_to(directories.project_root)}")


def load_dataset(path, config):
    try:
        if not config.load_data['select_columns']:
            if not config.load_data['load_params']:
                data = pd.read_csv(path)
            else:
                data = pd.read_csv(path, **config.load_data['load_params'])
        else:
            if not config.load_data['load_params']:
                data = pd.read_csv(path, usecols=config.load_data['select_columns']['use_features'])
            else:
                data = pd.read_csv(path, **config.load_data['load_params'], usecols=config.load_data['select_columns']['use_features'])
    except FileNotFoundError as e:
        raise DatasetNotFoundError(f"{path} is missing...") from e

    return data

def load_intermediary_dataset(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError as e:
        raise DatasetNotFoundError(f"{path} is missing...") from e
    return data

def save_model(model, *, path):
    return joblib.dump(model, path)


def save_metrics(metrics, *, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

def save_plots(plots, *, path):
    shap_values = plots['shap_values']
    shap_values.savefig(path/ 'shap.png', bbox_inches='tight')

def save_training_output(output, *, directory):
    model = output['model']
    metrics = output['metrics']
    plots = output['plots']
    save_model(model, path=directory / filenames.MODEL)
    save_metrics(metrics, path=directory / filenames.METRICS)
    save_plots(plots, path=directory)

def save_rai_training_output(output, *, directory):

    save_dataset(output[filenames.BIAS_RANK_ACTUAL], path=directory / f'{filenames.BIAS_RANK_ACTUAL}.csv')
    save_dataset(output[filenames.BIAS_RANK_PROXY], path=directory / f'{filenames.BIAS_RANK_PROXY}.csv')

def load_model(path):
    return joblib.load(path)


def load_prediction_data(path):
    return pd.read_json(path)

def save_predictions(predictions, *, path):
    with open(path, 'w') as f:
        json.dump(predictions.to_dict(), f)