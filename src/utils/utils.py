import logging

import pandas as pd
import numpy as np
from sklearn import preprocessing

from src.constants import *


logger = logging.getLogger(__name__)


def clean_age_income(data):

    logger.info(f"Starting individual/year records: {data.shape}")
    model_input = data[data[INCTOT] != 9999999]
    model_input = model_input[model_input[INCTOT] > 0]

    logger.info(f"Removing null income - remaining records:  {model_input.shape}")

    # model_input = model_input[(model_input[AGE] > 18) & (model_input[AGE] > 80)]
    model_input = model_input[(model_input[AGE] > 18)]

    logger.info(f"Removing ages <18 & >80 - remaining records:  {model_input.shape}")

    return (model_input)


def create_race_groupings(model_input):
    #### Group based on US Census race definitions

    # Group 1: Asian
    # Group 2: White, not Hispanic
    # Group 3: Hispanic (any race)
    # Group 4: Black

    # by group
    conditions = [
        (model_input['hispan'] == 1),
        (model_input['race'] == 4) | (model_input['race'] == 5) | (model_input['race'] == 6),
        (model_input['race'] == 1) & (model_input['hispan'] == 0),
        (model_input['race'] == 2)
    ]

    choices = ['Hispanic any race', 'Asian', 'White not Hispanic', 'Black']
    model_input['race_group'] = np.select(conditions, choices, default='Other race')
    model_input['non_white'] = 1 - np.where(model_input['race_group'] == 'White not Hispanic', 1, 0)
    model_input['hispanic'] = np.where(model_input['race_group'] == 'Hispanic any race', 1, 0)
    model_input['black'] = np.where(model_input['race_group'] == 'black', 1, 0)
    model_input['asian'] = 1 - np.where(model_input['race_group'] == 'Asian', 1, 0)
    model_input['other_race'] = np.where(model_input['race_group'] == 'Other race', 1, 0)


    ### Not super representative of the US - skews more white
    # print(model_input['RACE_Grp'].value_counts(normalize=True))

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


