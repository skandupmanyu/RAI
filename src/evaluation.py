import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, jaccard_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# import shap


def accuracy(fitted_model, X, y):
    predicted = fitted_model.predict(X)
    return accuracy_score(y, predicted)


def auc_score(fitted_model, X, y):
    y_pred_proba = fitted_model.predict_proba(X)[::, 1]
    return roc_auc_score(y, y_pred_proba)

def roc_curve(fitted_model, X, y):
    y_pred_proba = fitted_model.predict_proba(X)[::, 1]
    fpr, tpr, _ = roc_curve(y, y_pred_proba)

    plt.plot(fpr, tpr, label="Hispanic, auc=" + str(roc_auc_score(y, y_pred_proba)))
    plt.legend(loc=4)
    plt.show()
    return

# def shap_values(model, X_train):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(X_train)
#     shap.summary_plot(shap_values[1], X_train.astype("float"))
#     return shap_values


_METRICS = {
    'accuracy': accuracy,
    'auc_score': auc_score,
}



def evaluate_model(fitted_model, *, X, y):
    metrics = {name: func(fitted_model, X, y) for name, func in _METRICS.items()}
    return metrics