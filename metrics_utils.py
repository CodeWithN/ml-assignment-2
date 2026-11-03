
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def compute_metrics(model, X, y_true):
    """
    Evaluate a trained classification model (binary or multi-class).

    Precision, Recall and F1 use macro averaging so every class
    contributes equally regardless of how common it is.
    """

    y_true = np.asarray(y_true).ravel()

    if len(y_true) == 0:
        raise ValueError("The test dataset contains no target values.")

    model_classes = np.asarray(model.classes_)

    if len(model_classes) < 2:
        raise ValueError(
            "This project expects a classifier trained on at least two classes."
        )

    if not np.isin(y_true, model_classes).all():
        raise ValueError(
            "The test data contains labels that the model was not trained on."
        )

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "The test data must contain at least two classes to calculate AUC."
        )

    y_pred = np.asarray(model.predict(X)).ravel()

    if len(y_pred) != len(y_true):
        raise ValueError(
            "The number of predictions does not match the actual labels."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The model must support probability predictions to calculate AUC."
        )

    probabilities = model.predict_proba(X)

    accuracy = accuracy_score(y_true, y_pred)

    if len(model_classes) == 2:
        positive_index = np.where(model_classes == model_classes.max())[0][0]
        auc = roc_auc_score(y_true, probabilities[:, positive_index])
    else:
        auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=model_classes,
        )

    precision = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
    }
