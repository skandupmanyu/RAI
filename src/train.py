import logging
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

from src import directories
from src.RAI import ModelBiasRanker
from src.evaluation import evaluate_model, create_plots
from src.in_out import save_dataset
from src.constants import CROSS_TAB_METRICS, BIAS_ACTUAL, BIAS_PROXY, TPR_FPR

logger = logging.getLogger(__name__)

def train_proxy(model, model_data, config):
    X = model_data.drop(labels=[config.pg_target, config.rai_target], axis=1)

    y = model_data[config.pg_target]

    cat_features = []
    for cat in X.select_dtypes(exclude="number"):
        cat_features.append(cat)
        X[cat] = X[cat].astype("category").cat.codes.astype("category")

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=config.test_cutoff, random_state=config.split_random_size)
    #save training and test set
    save_dataset(X_train, path=directories.model / 'pg_x_train.csv')
    save_dataset(X_test, path=directories.model / 'pg_x_test.csv')
    save_dataset(y_train, path=directories.model / 'pg_y_train.csv')
    save_dataset(y_test, path=directories.model / 'pg_y_test.csv')

    logger.info(f"Train's shape: {X_train.shape}")
    logger.info(f"Test's shape: {X_test.shape}")

    logger.info(f"Training model...")
    fitted_model = model.fit(X_train,y_train, eval_set=(X_test,y_test), early_stopping_rounds=config.model['training_params']['early_stopping_rounds'], verbose=config.model['training_params']['verbose'])
    logger.info("Model trained.")


    train_metrics, test_metrics = (
        evaluate_model(fitted_model, X=features[0], y=features[1])
        for features in ((X_test, y_test),(X_train, y_train))
    )

    train_plots = (
        create_plots(fitted_model, X=X_train, y=y_train)
    )

    metrics_msg = "=" * 10 + " Metrics " + "=" * 10
    logger.info(metrics_msg)
    logger.info(f"Train: {train_metrics}")
    logger.info(f"Test: {test_metrics}")
    logger.info("=" * len(metrics_msg))

    # Assign a binary indicator to the highest 7.1% of model scores - these are the proxy group
    pred_proba = fitted_model.predict_proba(X)[::, 1]
    pg_rate = model_data[config.pg_target].value_counts(normalize=True)[1]
    pg_rate_thresh = np.percentile(pred_proba, 100*(1-pg_rate))

    # Count number of individauls labeled among 80K training set
    logger.info(f"Model count number of individuals labeled among training set: {np.where(pred_proba > pg_rate,1,0).sum()}.")

    # Perform cross-tab - accuracy is so so - 3.0 k 1,1 | 2.7 k 1,0, TPR < 70%
    crosstab_accuracy = pd.crosstab(model_data[config.pg_target],
                np.where(fitted_model.predict_proba(X)[::, 1] > pg_rate_thresh, 1, 0))  # .apply(lambda r: r/r.sum(), axis=1)

    tpr_fpr = create_cross_tab_metrics(crosstab_accuracy)

    save_dataset(crosstab_accuracy, path=directories.model/ CROSS_TAB_METRICS)
    save_dataset(tpr_fpr, path=directories.model / TPR_FPR)

    return {
        'model': fitted_model,
        'plots': train_plots,
        'metrics': {
            'train': train_metrics,
            'test': test_metrics
        }
    }

def train_rai(fitted_model, model, model_data, config):


    X = model_data.drop(labels=[config.pg_target, config.rai_target], axis=1)
    pred_proba = fitted_model.predict_proba(X)[::,1]

    pg_target_proxy = f'{config.pg_target}_proxy'

    pg_rate = model_data[config.pg_target].value_counts(normalize=True)[1]
    pg_rate_thresh = np.percentile(pred_proba, 100 * (1 - pg_rate))

    model_data[pg_target_proxy] = np.where(pred_proba > pg_rate_thresh, 1, 0)

    y_pg_proxy = model_data[pg_target_proxy]
    logger.info(
        f"{config.rai_target} distribution: {model_data[config.rai_target].value_counts(normalize=True)}.")


    #save dataframe historical bias
    bias_proxy = model_data[[pg_target_proxy, config.rai_target]].groupby([pg_target_proxy]).mean()
    bias_actual = model_data[[config.pg_target, config.rai_target]].groupby([config.pg_target]).mean()
    save_dataset(bias_proxy, path=directories.model / BIAS_ACTUAL)
    save_dataset(bias_actual, path=directories.model / BIAS_PROXY)

    # actual
    X_actual = model_data.drop([config.rai_target, pg_target_proxy, config.pg_target], axis=1)
    y_actual = model_data[config.rai_target]
    y_pg = model_data[config.pg_target]

    grid = {"n_estimators": [100, 200, 400],
            "max_depth": [4, 6, 8, 10]}

    clf_cv = GridSearchCV(model, grid, cv=5, scoring="roc_auc", refit=True, verbose=3, n_jobs=-1)
    clf_cv.fit(X_actual, y_actual)

    best_estimator = clf_cv.best_estimator_
    logging.info(f'The best parameters are: {clf_cv.best_params_}')

    bias_rank_actual = bias_ranker(X_actual, y_actual, y_pg, best_estimator,
                                   config.bias_ranker['pos_rate'], config.bias_ranker['bias_tolerance'])

    # proxy
    bias_rank_proxy = bias_ranker(X_actual, y_actual, y_pg_proxy, best_estimator,
                                  config.bias_ranker['pos_rate'], config.bias_ranker['bias_tolerance'])

    return {
        'bias_rank_actual': bias_rank_actual,
        'bias_rank_proxy': bias_rank_proxy
            }


def bias_ranker(X, y, y_pg, best_estimator, pos_rate, bias_tolerance):
    bias_ranker = ModelBiasRanker(best_estimator,
                                  pos_rate=pos_rate,
                                  bias_tolerance=bias_tolerance)

    bias_ranker.fit(X, y, y_pg)

    return bias_ranker.results_

def create_cross_tab_metrics(crosstab_accuracy: pd.DataFrame) -> pd.DataFrame:
    TPR = (crosstab_accuracy[1][1] / (crosstab_accuracy[1][1] + crosstab_accuracy[0][1])).round(2)
    FPR = (crosstab_accuracy[0][1] / (crosstab_accuracy[0][1] + crosstab_accuracy[0][0])).round(2)
    matrix = pd.DataFrame(index=['metrics'], data={'TPR': TPR * 100, 'FPR': FPR * 100})

    return matrix