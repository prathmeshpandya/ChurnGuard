"""Score a trained model on data it has never seen (the test set).

Metric cheat sheet (1 = churned is the "positive" class):
  accuracy  = % of all predictions that are right. Misleading when churners are rare.
  precision = of customers we FLAGGED as churners, % who really churned. (Few false alarms?)
  recall    = of customers who REALLY churned, % we caught.             (Few missed churners?)
  f1        = one number balancing precision and recall.
  roc_auc   = how well the model RANKS churners above non-churners, at any threshold.
              0.5 = random guessing, 1.0 = perfect.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X, y, threshold: float = 0.5) -> dict:
    """Return the main metrics as a plain dict (easy to print and save as JSON)."""
    # predict_proba gives [P(stays), P(churns)] per row; [:, 1] keeps the churn probability.
    churn_probability = model.predict_proba(X)[:, 1]
    # Turn probabilities into yes/no decisions. 0.5 is the default; Day 5 tunes this.
    predicted = (churn_probability >= threshold).astype(int)

    # ravel() flattens the 2x2 table into its 4 cells, in this fixed order.
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy_score(y, predicted),
        # zero_division=0: if the model never flags anyone, precision is 0 instead of an error.
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, churn_probability),
        # Confusion matrix cells:
        "true_negatives": int(tn),   # said "stays", really stayed       (correct)
        "false_positives": int(fp),  # said "churns", really stayed      (false alarm)
        "false_negatives": int(fn),  # said "stays", really churned      (missed churner)
        "true_positives": int(tp),   # said "churns", really churned     (caught)
    }


def plot_confusion_matrix(model, X, y, title: str, ax: plt.Axes) -> None:
    """Draw a model's confusion matrix on the given chart area."""
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=["stays", "churns"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
