import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency


class DataBiasChecker:
    def __init__(
        self,
        pvalue_threshold=0.1,
        test_type="z-test",
        is_2_sided=False,
    ) -> None:
        self.pvalue_threshold = pvalue_threshold
        self.is_2_sided = is_2_sided
        self.test_type = test_type

    def fit(self, pg, y):
        if self.test_type == "z-test":
            self._stat_test_z(pg, y)
        elif self.test_type == "categorical":
            self._stat_test_z(pg, y)
        elif self.test_type == "welch":
            self._stat_test_welch(pg, y)
        else:
            self._stat_test_z(pg, y)
        return self

    def _stat_test_categoric(self, pg, y):
        ctable = pd.crosstab(pg, y)
        self.chi2, self.p_value_, self.dof, self.ex = chi2_contingency(
            ctable, correction=False
        )
        if self.p_value_ < self.pvalue_threshold:
            self.biased_ = True
        else:
            self.biased_ = False
        return self.biased_, self.p_value_

    def _stat_test_z(self, pg, y):
        class_names = pg.unique()
        self.historic_crosstab_ = pd.crosstab(pg, y).apply(
            lambda r: r / r.sum(), axis=1
        )
        var1 = y[pg == class_names[0]]
        var2 = y[pg == class_names[1]]
        # equal_val = False makes it a welch test
        self.statistics, self.p_value_ = stats.ttest_ind(var1, var2, equal_var=True)
        if not self.is_2_sided:
            self.p_value_ = self.p_value_ / 2
        if self.p_value_ < self.pvalue_threshold:
            self.biased_ = True
        else:
            self.biased_ = False
        return self.biased_, self.p_value_

    def _stat_test_welch(self, pg, y):
        class_names = pg.unique()
        var1 = y[pg == class_names[0]]
        var2 = y[pg == class_names[1]]
        # equal_val = False makes it a welch test
        self.statistics, self.p_value_ = stats.ttest_ind(var1, var2, equal_var=False)
        if not self.is_2_sided:
            self.p_value_ = self.p_value_ / 2
        if self.p_value_ < self.pvalue_threshold:
            self.biased_ = True
        else:
            self.biased_ = False
        return self.biased_, self.p_value_

    def get_params(self):
        return {
            "pvalue_threshold": self.pvalue_threshold,
            "test_type": self.test_type,
            "is_2_sided": self.is_2_sided,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
