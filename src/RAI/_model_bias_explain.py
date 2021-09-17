"""
Last updated 6/25/2020
This function trains a groups the model predicted outcomes by the
the predicted probabilities and default model binary predictions, as well as the
outcome and protected group arrays
"""
import numpy as np
import pandas as pd
from sklearn import metrics


class ModelBiasHandler:
    def __init__(self, clf, target_rate, bias_detect_thresh) -> None:
        self.target_rate = target_rate
        self.bias_detect_thresh = bias_detect_thresh
        self.clf = clf

    def fit(self, X, y, pg):

        # Output model predictions and probabilities
        preds_proba_raw = self.clf.predict_proba(X)
        y_pred_proba = np.array(preds_proba_raw)[:, 1]
        y_pred_naive = self.clf.predict(X)

        self.scenario_preds = pd.DataFrame()
        self.scenario_preds["y_true"] = y
        self.scenario_preds["protected_group"] = pg
        self.scenario_preds["preds_proba"] = y_pred_proba
        self.scenario_preds["pred_naive"] = y_pred_naive

        # Run get bias metrics function
        self._run_naive_scenario()
        self._run_best_thresh_scenario()
        self._run_hist_scenario()
        self._run_demog_scenario()

        return self

    def predict(self, X, pg, scenario="demog"):
        preds_proba_raw = self.clf.predict_proba(X)
        y_pred_proba = np.array(preds_proba_raw)[:, 1]

        if scenario == "naive":
            y_pred = self.clf.predict(X)
        if scenario == "thresh_best":
            y_pred = (y_pred_proba > self.thresh_best_).astype("int")
        if scenario == "historic":
            y_pred = np.where(
                pg == 1,
                (y_pred_proba >= self.thresh_hist_pg_).astype(int),
                (y_pred_proba >= self.thresh_hist_non_pg_).astype(int),
            )
        if scenario == "demog":
            y_pred = np.where(
                pg == 1,
                (y_pred_proba > self.thresh_demog_pg_).astype(int),
                (y_pred_proba > self.thresh_demog_non_pg_).astype(int),
            )
        return y_pred

    def summary(self):

        scenario_names = ["naive", "th_best", "hist", "demog"]
        metrics = [
            "bias_test",
            "bias_index",
            "acc",
            "TP",
            "FN",
            "TN",
            "FP",
            "non_pg_rate",
        ]
        fairness_scenarios = pd.DataFrame(columns=metrics, index=scenario_names)
        for scenario_name in scenario_names:
            for metric in metrics:
                fairness_scenarios.loc[scenario_name, metric] = getattr(
                    self, f"{metric}_{scenario_name}_"
                )

        # Formatting
        fairness_scenarios.columns = [
            "Bias_Test",
            "Bias_Index",
            "Accuracy",
            "TP",
            "FN",
            "TN",
            "FP",
            "Non_PG_Outcome_Rate",
        ]
        fairness_scenarios["PG_Outcome_Rate"] = (
            (
                fairness_scenarios["Non_PG_Outcome_Rate"]
                * fairness_scenarios["Bias_Index"]
            )
            .astype(float)
            .round(4)
        )

        fairness_scenarios.index = [
            "Naive",
            "Threshold Best",
            "Historic Parity",
            "Demographic Parity",
        ]

        return fairness_scenarios

    def get_params(self):
        return {
            "clf": self.clf,
            "target_rate": self.target_rate,
            "bias_detect_thresh": self.bias_detect_thresh,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _run_naive_scenario(self):

        scenario_name = "naive"
        return self._get_bias_metrics(scenario_name)

    def _run_best_thresh_scenario(self):

        scenario_name = "th_best"
        pred_col_name = f"pred_{scenario_name}"
        # Force threshold to be equal to the population outcome rate (bad loan)
        force_thresh = pd.DataFrame(self.scenario_preds["preds_proba"]).quantile(
            self.target_rate
        )
        threshold = force_thresh[0]
        self.thresh_best_ = threshold
        self.scenario_preds[pred_col_name] = (
            self.scenario_preds["preds_proba"] > self.thresh_best_
        ).astype("int")

        return self._get_bias_metrics(scenario_name)

    def _run_hist_scenario(self):

        scenario_name = "hist"
        pred_col_name = f"pred_{scenario_name}"
        # create separate DFs for PG and non-PG groups
        X_test_pg = self.scenario_preds[self.scenario_preds["protected_group"] == 1]
        X_test_non_pg = self.scenario_preds[self.scenario_preds["protected_group"] != 1]
        N_pg = X_test_pg.shape[0]  # number of instances in pg group
        N_npg = X_test_non_pg.shape[0]  # number of instances in non-pg group

        # Force thresholds to be equal to the population historical outcome rate
        pg_baseline = (
            self.scenario_preds[["protected_group", "y_true"]]
            .groupby(["protected_group"], as_index=False)
            .mean()
        )
        # compute the historic bias index as positive_rate_pg / positive_rate_non_pg
        hist_bias_index = pg_baseline["y_true"][1] / pg_baseline["y_true"][0]

        # computing desired size of positives for pg & non-pg iff we want the new
        # bias index to be equal to historic one
        npg = ((N_pg + N_npg) * (1 - self.target_rate)) / (
            hist_bias_index * N_pg / N_npg + 1
        )
        pg = npg * hist_bias_index * N_pg / N_npg

        # computing desired positive rates
        positive_rate_pg = pg / float(N_pg)
        positive_rate_npg = npg / float(N_npg)

        # chose threshold for pg & non-pg to get the desired positive rates

        pg_thresh = pd.DataFrame(X_test_pg["preds_proba"]).quantile(
            1 - positive_rate_pg
        )
        non_pg_thresh = pd.DataFrame(X_test_non_pg["preds_proba"]).quantile(
            1 - positive_rate_npg
        )

        # Maybe there is a cleaner way to do this, data type issue so include this
        # step
        threshold_pg = pg_thresh[0]
        threshold_non_pg = non_pg_thresh[0]
        self.thresh_hist_pg_ = threshold_pg
        self.thresh_hist_non_pg_ = threshold_non_pg

        # Create predictions using model output probabilities
        X_test_pg["predictions_hist"] = (
            X_test_pg["preds_proba"] >= self.thresh_hist_pg_
        ).astype("int")
        X_test_non_pg["predictions_hist"] = (
            X_test_non_pg["preds_proba"] >= self.thresh_hist_non_pg_
        ).astype("int")

        # Combine DFs
        X_test_adj = X_test_pg.append(X_test_non_pg)

        # Create preds
        self.scenario_preds[pred_col_name] = X_test_adj["predictions_hist"]

        return self._get_bias_metrics(scenario_name)

    def _run_demog_scenario(self):

        scenario_name = "demog"
        pred_col_name = f"pred_{scenario_name}"
        # create separate DFs for PG and non-PG groups
        X_test_pg = self.scenario_preds[self.scenario_preds["protected_group"] == 1]
        X_test_non_pg = self.scenario_preds[self.scenario_preds["protected_group"] != 1]
        # Force thresholds to be equal among PG and non-PG
        pg_thresh_eq = pd.DataFrame(X_test_pg["preds_proba"]).quantile(self.target_rate)
        non_pg_thresh_eq = pd.DataFrame(X_test_non_pg["preds_proba"]).quantile(
            self.target_rate
        )

        # Maybe there is a cleaner way to do this, data type issue so include this
        # step
        threshold_pg = pg_thresh_eq[0]
        threshold_non_pg = non_pg_thresh_eq[0]
        self.thresh_demog_pg_ = threshold_pg
        self.thresh_demog_non_pg_ = threshold_non_pg

        # Create predictions using model output probabilities
        X_test_pg["preds_demographic"] = (
            X_test_pg["preds_proba"] > self.thresh_demog_pg_
        ).astype("int")
        X_test_non_pg["preds_demographic"] = (
            X_test_non_pg["preds_proba"] > self.thresh_demog_non_pg_
        ).astype("int")

        # Combine DFs
        X_test_adj = X_test_pg.append(X_test_non_pg)

        # Create preds
        self.scenario_preds[pred_col_name] = X_test_adj["preds_demographic"]

        return self._get_bias_metrics(scenario_name)

    def _get_bias_metrics(self, scenario_name):

        pred_col_name = f"pred_{scenario_name}"
        # Create pivot working class - in population overall (train and test)
        pg_compare = (
            self.scenario_preds[["protected_group", pred_col_name]]
            .groupby(["protected_group"], as_index=False)
            .mean()
        )

        bias_index = pg_compare[pred_col_name][1] / pg_compare[pred_col_name][0]

        if abs(round(bias_index - 1, 3)) > self.bias_detect_thresh:
            bias_test = "Fail"
        else:
            bias_test = "Pass"

        # Profitability
        non_pg_rate = pg_compare[pred_col_name][0]
        acc = metrics.accuracy_score(
            self.scenario_preds[pred_col_name], self.scenario_preds["y_true"]
        )

        # Confusion Matrix Values
        cm = pd.crosstab(
            self.scenario_preds[pred_col_name], self.scenario_preds["y_true"]
        ).apply(lambda r: r / r.sum(), axis=1)
        TN = round(cm[0][0], 4)
        FN = round(cm[0][1], 4)
        FP = round(cm[1][0], 4)
        TP = round(cm[1][1], 4)
        bias_index = round(bias_index, 4)
        acc = round(acc, 4)
        non_pg_rate = round(non_pg_rate, 4)

        # Combine lists into string output
        params = {
            f"bias_test_{scenario_name}_": bias_test,
            f"bias_index_{scenario_name}_": bias_index,
            f"acc_{scenario_name}_": acc,
            f"TP_{scenario_name}_": TP,
            f"FN_{scenario_name}_": FN,
            f"TN_{scenario_name}_": TN,
            f"FP_{scenario_name}_": FP,
            f"non_pg_rate_{scenario_name}_": non_pg_rate,
        }
        self.set_params(**params)

        return params.values()
