import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing

__all__ = ['build_features_set']

logger = logging.getLogger(__name__)



def build_features_set(model_sample, num_vars, cat_vars, config):
    logger.info("Building features set.")
    if list(set(model_sample.columns) & set(cat_vars)):
        # This is an internal function which one-hot encodes all categorical and normalizes all numeric variables
        model_data_norm = normalize_num(model_sample, num_vars)
        model_data_cat = pd.get_dummies(model_sample[cat_vars], columns=cat_vars)
        model_data_num = model_data_norm[num_vars]

        # temp fix issue with multiple indices, creating more rows
        model_data_num.index = model_sample.index
        model_data_norm.index = model_sample.index
        model_data_cat.index = model_sample.index

        # Join
        model_data_norm['id'] = model_data_norm.index
        model_data = pd.concat([model_data_norm[[config.pg_target,config.rai_target, "id"]], model_data_cat, model_data_num], axis=1)
        model_data = model_data.drop(['id'], axis=1)
        try:
            model_data = model_data.drop(cat_vars, axis=1)
        except:
            logger.info(f'The following categorical variables {cat_vars} are filled with numerical values')

    else:
        model_data = normalize_num(model_sample, num_vars)
    # Print shape of final machine-learning ready data frame
    print("Model data shape: ", model_data.shape)
    logger.info(
        f"""Successfully built feature set with {len(model_data)} rows and """
        f"""{model_data.shape[1]} columns."""
    )

    return model_data

def build_features_set_prediction(model_sample, num_vars, cat_vars, config):
    logger.info("Building features set.")
    if list(set(model_sample.columns) & set(cat_vars)):
        # This is an internal function which one-hot encodes all categorical and normalizes all numeric variables
        model_data_norm = normalize_num(model_sample, num_vars)
        model_data_cat = pd.get_dummies(model_sample[cat_vars], columns=cat_vars)
        model_data_num = model_data_norm[num_vars]

        # temp fix issue with multiple indices, creating more rows
        model_data_num.index = model_sample.index
        model_data_norm.index = model_sample.index
        model_data_cat.index = model_sample.index

        # Join
        model_data_norm['id'] = model_data_norm.index
        model_data = pd.concat([model_data_norm[["id"]], model_data_cat, model_data_num], axis=1)
        model_data = model_data.drop('id', axis=1)
    else:
        model_data = normalize_num(model_sample, num_vars)
    # Print shape of final machine-learning ready data frame
    print("Model data shape: ", model_data.shape)
    logger.info(
        f"""Successfully built feature set with {len(model_data)} rows and """
        f"""{model_data.shape[1]} columns."""
    )

    return model_data

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