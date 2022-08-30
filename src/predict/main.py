import logging
import pandas as pd
import numpy as np

from src.constants.filenames import PREDICTIONS
from src.features import build_features_set_prediction
from src.in_out import (
    load_model,
    load_prediction_data,
    save_predictions as _save_predictions
)
from src.utils.utils import (clean_column_names, create_race_groupings_ipums
)
from src.data import (msa_extractor,
)

from src import config, directories
from src.constants import MODEL, AGE


logger = logging.getLogger(__name__)


def predict(config):
    data = _get_prediction_data(config)
    model = _load_model()
    predictions = _predict(data, config, model=model)
    return predictions

def _get_prediction_data(config):
    data = _load_prediction_data(config)
    return data

def _load_model():
    path = directories.model / MODEL
    return load_model(path)

def _load_prediction_data(config):
    path = directories.input / config.prediction_data
    return load_prediction_data(path)

def _predict(data, config, *, model):
    data = data.rename(clean_column_names(data.columns), axis=1)

    data = create_race_groupings_ipums(data, config)
    targets = [config.pg_target, AGE]
    if config.use_msa:
        data, numerical_columns_created = msa_extractor(data, data, targets, config)

    num_vars = data._get_numeric_data().columns.tolist()
    num_vars = list(set(num_vars) - set([config.pg_target, config.rai_target]))
    cat_vars = config.features['categorical']

    features = build_features_set_prediction(data, num_vars, cat_vars, config)
    features = features.reindex(sorted(features.columns), axis=1)

    pred_proba = model.predict_proba(features)[::,1]

    pg_rate = config.proportion_pg_traget_over_population
    pg_rate_thresh = np.percentile(pred_proba, 100 * (1 - pg_rate))
    data['prediction'] = (pred_proba >= pg_rate_thresh).astype(int)
    return data[['prediction']]

def save_predictions(predictions):
    path = directories.artefacts_dir / config.model_name / PREDICTIONS
    _save_predictions(predictions, path=path)