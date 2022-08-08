import logging

import pandas as pd
import numpy as np
from sklearn import preprocessing

from src.constants import *


logger = logging.getLogger(__name__)


def clean_age_income(data):
    logger.info(f"Removing ages <18 - remaining records:  {data.shape}")
    data = data[(data[AGE] >= 18)]
    logger.info("Filter down to working age (18-80) records for which we have income")
    data = data[data[INCTOT] != 9999999]
    data = data[data[INCTOT] > 0]
    return (data)

def strip_whitespace(data):
    return data.replace(r"^ +| +$", r"", regex=True)

def create_race_groupings_adult(model_input, config):

    race = {}
    race['white'] = np.where(model_input['race'] == 'White', 1, 0)
    race['hispanic'] = np.where(model_input['race'] == 'Hispanic any race', 1, 0)
    race['amer_indian'] = np.where(model_input['race'] == 'Amer-Indian-Eskimo', 1, 0)
    race['black'] = np.where(model_input['race'] == 'Black', 1, 0)
    race['asian'] = np.where(model_input['race'] == 'Asian-Pac-Islander', 1, 0)
    race['other_race'] = np.where(model_input['race'] == 'Other', 1, 0)

    model_input[config.pg_target] = race[config.pg_target]
    model_input = model_input.drop('race',axis=1)
    return model_input

def create_race_groupings_ipums(model_input, config):
    #### Group based on US Census race definitions

    # Group 1: Asian
    # Group 2: White, not Hispanic
    # Group 3: Hispanic (any race)
    # Group 4: Black

    # by group
    conditions = [
        (model_input['hispan'] == 1),
        (model_input['race'] == 2),
        (model_input['race'] == 4) | (model_input['race'] == 5) | (model_input['race'] == 6),
        (model_input['race'] == 1) & (model_input['hispan'] == 0)

    ]

    choices = ['Hispanic any race', 'Black', 'Asian', 'White not Hispanic']

    group = {}
    model_input['race_group'] = np.select(conditions, choices, default='Other race')
    group['non_white'] = 1 - np.where(model_input['race_group'] == 'White not Hispanic', 1, 0)
    group['hispanic'] = np.where(model_input['race_group'] == 'Hispanic any race', 1, 0)
    group['black'] = np.where(model_input['race_group'] == 'Black', 1, 0)
    group['asian'] = 1 - np.where(model_input['race_group'] == 'Asian', 1, 0)
    group['other_race'] = np.where(model_input['race_group'] == 'Other race', 1, 0)

    model_input[config.pg_target] = group[config.pg_target]
    model_input = model_input.drop(['race', 'hispan', 'race_group'],axis=1)
    model_input[config.rai_target] = np.where(model_input[config.rai_target] > np.percentile(model_input[config.rai_target], config.threshold_rai_target), 1, 0)

    return model_input


def clean_column_names(column_list):
    """Simple string cleaning rules for columns."""
    new_column_list = [
        (
            col.lower()
            .strip()
            .replace("  ", "_")
            .replace(r"/", "_")
            .replace(r"\n", "_")
            .replace(r"\\", "_")
            .replace(r"\t", "_")
            .replace(" ", "_")
            .replace("^", "")
        )
        for col in column_list
    ]
    return dict(zip(column_list, new_column_list))

def normalize_num(df, num_cols):
    """
    This function normalizes numeric variables

    Args:
        df: pandas df with numeric columns to normalize
        num_cols: list of numeric column names

    Result:
        Returns same dataframe with normalized data
    """

    x = df[num_cols].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_num = pd.DataFrame(x_scaled)
    df_num.columns = num_cols
    df_num = df_num.reset_index(drop=True)

    df_drop = df.drop(num_cols, axis=1)
    df_drop = df_drop.reset_index(drop=True)
    df_final = pd.concat([df_drop, df_num], axis=1)

    return df_final


