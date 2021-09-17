import pandas as pd

from ._model_bias_explain import ModelBiasHandler


class ModelBiasRanker:
    def __init__(self, clf, pos_rate, bias_tolerance) -> None:
        self.pos_rate = pos_rate
        self.bias_tolerance = bias_tolerance
        self.clf = clf

    def fit(self, X, y, pg):
        methods_nm = [
            "Naive",
            "Threshold Best",
            "Historic Parity",
            "Demographic Parity",
        ]
        metrics_nm = [
            "Bias Test",
            "Bias Index",
            "Accuracy",
            "TP",
            "FN",
            "TN",
            "FP",
            "Non PG Positive Rate",
            "PG Positive Rate",
        ]
        self.results_ = pd.DataFrame(columns=metrics_nm, index=methods_nm)

        bias_handler = ModelBiasHandler(
            self.clf, self.pos_rate, self.bias_tolerance, "naive"
        )
        bias_handler.fit(X, y, pg)
        self._assign_attr_to_summary(bias_handler, methods_nm[0], metrics_nm)

        bias_handler = ModelBiasHandler(
            self.clf, self.pos_rate, self.bias_tolerance, "thresh_best"
        )
        bias_handler.fit(X, y, pg)
        self._assign_attr_to_summary(bias_handler, methods_nm[1], metrics_nm)

        bias_handler = ModelBiasHandler(
            self.clf, self.pos_rate, self.bias_tolerance, "historic"
        )
        bias_handler.fit(X, y, pg)
        self._assign_attr_to_summary(bias_handler, methods_nm[2], metrics_nm)

        bias_handler = ModelBiasHandler(
            self.clf, self.pos_rate, self.bias_tolerance, "demog_parity"
        )
        bias_handler.fit(X, y, pg)
        self._assign_attr_to_summary(bias_handler, methods_nm[3], metrics_nm)

        return self

    def _assign_attr_to_summary(self, bias_handler, method_nm, metrics_nm):
        self.results_.loc[method_nm, metrics_nm[0]] = bias_handler.bias_test_
        self.results_.loc[method_nm, metrics_nm[1]] = round(bias_handler.bias_index_, 4)
        self.results_.loc[method_nm, metrics_nm[2]] = round(bias_handler.acc_, 4)
        self.results_.loc[method_nm, metrics_nm[3]] = round(bias_handler.TP_, 4)
        self.results_.loc[method_nm, metrics_nm[4]] = round(bias_handler.FN_, 4)
        self.results_.loc[method_nm, metrics_nm[5]] = round(bias_handler.TN_, 4)
        self.results_.loc[method_nm, metrics_nm[6]] = round(bias_handler.FP_, 4)
        self.results_.loc[method_nm, metrics_nm[7]] = round(
            bias_handler.non_pg_rate_, 4
        )
        self.results_.loc[method_nm, metrics_nm[8]] = round(bias_handler.pg_rate_, 4)
        return
