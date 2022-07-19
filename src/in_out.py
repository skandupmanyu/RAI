import json
import logging

import joblib
import pandas as pd

from src.config.directories import directories
from src.constants import filenames
from src.constants.columns import COLUMNS
from src.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)

def get_data():

    logger.info("Getting dataset.")

    ## Download input file from BCG Egnyte - RAI folder
    ## https://bcg01.egnyte.com/navigate/folder/881ba5c2-8f6e-4eb6-b940-848faa78dee9
    input_path = directories.raw_data_dir / filenames.INPUT_DATASET,
    dataset = load_dataset(input_path[0],
                          compression='gzip',
                          header=0,
                          sep=',',
                          quotechar='"',
                          error_bad_lines=False,
                          usecols=COLUMNS)
    logger.info(f"Dataset loaded with {len(dataset)} rows.")

    return dataset

def save_dataset(dataset, *, path):
    dataset.to_csv(path, index=False)
    logger.info(f"Dataset saved at {path.relative_to(directories.project_root)}")


def load_dataset(path, **params):
    try:
        data = pd.read_csv(path, **params)
    except FileNotFoundError as e:
        raise DatasetNotFoundError(f"{path} is missing...") from e

    return data


def save_model(model, *, path):
    return joblib.dump(model, path)


def save_metrics(metrics, *, path):
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_training_output(output, *, directory):
    model = output['model']
    metrics = output['metrics']
    save_model(model, path=directory / filenames.MODEL)
    save_metrics(metrics, path=directory / filenames.METRICS)

def save_rai_training_output(output, *, directory):

    save_dataset(output[filenames.BIAS_RANK_ACTUAL], path=directory / f'{filenames.BIAS_RANK_ACTUAL}.csv')
    save_dataset(output[filenames.BIAS_RANK_ACTUAL], path=directory / f'{filenames.BIAS_RANK_PROXY}.csv')

def load_model(path):
    return joblib.load(path)


def load_prediction_data(path):
    with open(path) as fp:
        return json.load(fp)

def save_predictions(predictions, *, path):
    with open(path) as f:
        json.dump(predictions, f)