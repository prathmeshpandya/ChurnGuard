"""Build complete models: preprocessor + classifier in one scikit-learn Pipeline.

Why one Pipeline? Calling model.fit(X_train, y_train) then fits the preprocessor AND the
classifier on training data only (no leakage), and model.predict(raw_rows) works on raw
columns directly. That is exactly what the API on Day 6 needs.
"""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churnguard.config import RANDOM_STATE, DatasetConfig
from churnguard.features import build_preprocessor

# The model kinds we can build. Day 4 adds random forest and XGBoost here.
MODEL_KINDS = ("dummy", "logreg", "logreg_balanced")


def _make_classifier(kind: str):
    if kind == "dummy":
        # Always predicts the most common class ("stays"). Learns nothing about customers.
        # Any real model must beat this, or it is useless.
        return DummyClassifier(strategy="most_frequent")
    if kind == "logreg":
        # Logistic regression: learns one weight per input column, adds them up, and squashes
        # the total into a probability between 0 and 1 of churning.
        # max_iter=1000 gives the solver enough steps to finish learning.
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    if kind == "logreg_balanced":
        # Same model, but mistakes on churners (the rarer class) count more during training,
        # in proportion to how rare they are. It catches more churners, with more false alarms.
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        )
    raise ValueError(f"Unknown model kind {kind!r}. Choose one of: {MODEL_KINDS}")


def build_model(config: DatasetConfig, kind: str) -> Pipeline:
    """Return an untrained Pipeline: preprocess the columns, then classify."""
    return Pipeline([
        ("preprocess", build_preprocessor(config)),
        ("classifier", _make_classifier(kind)),
    ])
