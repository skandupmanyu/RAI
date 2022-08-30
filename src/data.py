import logging
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from src.constants import *
from src.constants import *
from src.features import build_features_set
from src.in_out import *
from src.utils.utils import (clean_age_income,
                             create_race_groupings_adult,
                             create_race_groupings_ipums,
                             clean_column_names,
                             strip_whitespace)



# Python settings
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

logger = logging.getLogger(__name__)

def build_dataset(config):
    logger.info("Loading raw dataset.")
    dataset = get_data(config)
    dataset = dataset.rename(clean_column_names(dataset.columns), axis=1)

    logger.info(f"Select training data up to {config.latest_year_dataset}")
    logger.info(f"Previous shape {dataset.shape}")
    dataset = dataset[dataset[YEAR] <= config.latest_year_dataset]
    logger.info(f"New shape {dataset.shape}")

    logger.info("Filter down to working age (18-80) records for which we have income")
    dataset = clean_age_income(dataset)

    logger.info(f"Take sample and build model input dataframe")
    dataset = dataset.sample(n=config.sample_size,
                                  random_state=config.random_state_sample)

    # logger.info("Strip whitespaces")
    # dataset = strip_whitespace(dataset)

    logger.info(f"Create a binary target variables")
    dataset = create_race_groupings_ipums(dataset, config)
    logger.info(f"Select only historical data for training")
    dataset_historical = dataset[dataset[YEAR] < config.latest_year_dataset]
    dataset_historical = dataset_historical.drop(YEAR, axis=1)
    dataset = dataset.drop(YEAR, axis=1)

    targets = [config.pg_target, AGE]

    if config.use_msa:
        dataset, numerical_columns_created = msa_extractor(dataset, dataset_historical, targets, config)

    num_vars = dataset._get_numeric_data().columns.tolist()
    num_vars = list(set(num_vars) - set([config.pg_target, config.rai_target]))
    # cat_vars = dataset.select_dtypes(include=['object']).columns.tolist()
    cat_vars = config.features['categorical']
    dataset_feat = build_features_set(dataset, num_vars, cat_vars, config)
    dataset_feat = dataset_feat.reindex(sorted(dataset_feat.columns), axis=1)

    return dataset_feat


def msa_extractor(model_input, model_input_historical, targets, config):
    summary_df_yr_msa = pd.DataFrame(model_input_historical.groupby(MET2013)[targets].mean())
    summary_df_yr_msa = summary_df_yr_msa.reset_index()
    summary_df_yr_msa.columns = [MET2013] + [f'msa_{x}_density' if x==config.pg_target else f'msa_{x}' for x in targets]

    model_input = model_input.merge(summary_df_yr_msa, on=MET2013)
    numerical_columns_created = summary_df_yr_msa.columns[1:].to_list()
    model_input = model_input.drop('met2013', axis=1)

    return model_input, numerical_columns_created