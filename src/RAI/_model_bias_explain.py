import numpy as np
import pandas as pd
from sklearn import metrics


class ModelBiasHandler:
    def __init__(self, clf, pos_rate, bias_tolerance, method) -> None:
        self.pos_rate = pos_rate
        self.bias_tolerance = bias_tolerance
        self.clf = clf
        self.method = method

    def fit(self, X, y, pg):
        # Run relevant fit function
        if self.method == "naive":
            self._fit_naive()
        if self.method == "thresh_best":
            self._fit_thresh_best(X)
        if self.method == "historic":
            self._fit_historic(X, y, pg)
        if self.method == "demog_parity":
            self._fit_demog(X, pg)
        # Get bias and output metrics
        self._get_bias_metrics(X, y, pg)
        return self

    def _fit_naive(self):
        self.thresh_pg_ = 0.5
        self.thresh_non_pg_ = 0.5
        return

    def _fit_thresh_best(self, X):
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        self.thresh_pg_ = np.quantile(y_pred_proba, 1 - self.pos_rate)
        self.thresh_non_pg_ = self.thresh_pg_
        return

    def _fit_historic(self, X, y, pg):
        N_pg = (pg == 1).sum()
        N_npg = (pg != 1).sum()
        hist_bias_index = self.compute_bias_index(y, pg)
        pos_rate_pg, pos_rate_npg = self._compute_pos_rates(
            N_pg, N_npg, hist_bias_index, self.pos_rate
        )
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        self.thresh_pg_ = np.quantile(y_pred_proba[pg == 1], 1 - pos_rate_pg)
        self.thresh_non_pg_ = np.quantile(y_pred_proba[pg != 1], 1 - pos_rate_npg)
        return

    def _fit_demog(self, X, pg):
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        self.thresh_pg_ = np.quantile(y_pred_proba[pg == 1], 1 - self.pos_rate)
        self.thresh_non_pg_ = np.quantile(y_pred_proba[pg != 1], 1 - self.pos_rate)
        return

    def _get_bias_metrics(self, X, y, pg):
        y_pred = self.predict(X, pg)
        self.bias_index_ = self.compute_bias_index(y_pred, pg)
        if abs(self.bias_index_ - 1) > self.bias_tolerance:
            self.bias_test_ = "Fail"
        else:
            self.bias_test_ = "Pass"
        # Accuracy tradeoff
        self.non_pg_rate_ = y_pred[pg != 1].mean()
        self.pg_rate_ = self.non_pg_rate_ * self.bias_index_
        self.acc_ = metrics.accuracy_score(y_pred, y)
        # Confusion Matrix
        cm = pd.crosstab(y_pred, y).apply(lambda r: r / r.sum(), axis=1)
        try:
            self.TN_ = cm[0][0]
        except:
            self.TN_ = np.NAN
        try:
            self.FN_ = cm[0][1]
        except:
            self.FN_ = np.NAN
        try:
            self.FP_ = cm[1][0]
        except:
            self.FP_ = np.NAN
        try:
            self.TP_ = cm[1][1]
        except:
            self.TP_ = np.NAN
    def predict(self, X, pg):
        y_pred_proba = self.clf.predict_proba(X)[:, 1]
        y_pred = np.where(
            pg == 1,
            (y_pred_proba >= self.thresh_pg_).astype(int),
            (y_pred_proba >= self.thresh_non_pg_).astype(int),
        )
        return y_pred

    def get_params(self):
        return {
            "clf": self.clf,
            "pos_rate": self.pos_rate,
            "bias_tolerance": self.bias_tolerance,
            "method": self.method,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @staticmethod
    def _compute_pos_rates(N_pg, N_npg, bias_index, pos_rate):
        npg = ((N_pg + N_npg) * pos_rate) / (bias_index * N_pg / N_npg + 1)
        pg = npg * bias_index * N_pg / N_npg
        pos_rate_pg = pg / float(N_pg)
        pos_rate_npg = npg / float(N_npg)
        return pos_rate_pg, pos_rate_npg

    @staticmethod
    def compute_bias_index(y, pg):
        return y[pg == 1].mean() / y[pg != 1].mean()
