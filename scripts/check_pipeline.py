"""Run load -> clean -> split -> preprocess on every dataset and print what came out.

    uv run python scripts/check_pipeline.py
"""

import numpy as np

from churnguard.config import DATASETS
from churnguard.data import load_dataset
from churnguard.features import build_preprocessor


def check(name: str) -> None:
    config = DATASETS[name]
    X_train, X_test, y_train, y_test = load_dataset(name)

    print(f"\n=== {name.upper()} ===")
    print(f"Train: {X_train.shape[0]:>5} rows | churn rate {y_train.mean():.1%}")
    print(f"Test:  {X_test.shape[0]:>5} rows | churn rate {y_test.mean():.1%}")

    preprocessor = build_preprocessor(config)
    # fit_transform on TRAIN: learns medians, scaling and categories from training data only.
    train_matrix = preprocessor.fit_transform(X_train)
    # transform on TEST: reuses what was learned from train. Never fit on test.
    test_matrix = preprocessor.transform(X_test)

    print(f"Columns in: {X_train.shape[1]}  ->  numbers out: {train_matrix.shape[1]}")
    print(f"Any missing values after preprocessing? {bool(np.isnan(train_matrix).any())}")
    print("Test matrix shape:", test_matrix.shape)

    names = preprocessor.get_feature_names_out()
    print("First 8 model inputs:", list(names[:8]))


if __name__ == "__main__":
    for dataset_name in DATASETS:
        check(dataset_name)
