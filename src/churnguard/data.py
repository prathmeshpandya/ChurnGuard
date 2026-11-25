"""Load, clean and split a churn dataset. Works for any dataset described in config.py."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import RANDOM_STATE, TEST_SIZE, DatasetConfig, get_config


def load_raw(config: DatasetConfig) -> pd.DataFrame:
    """Read the dataset's CSV exactly as downloaded."""
    if not config.csv_path.exists():
        raise FileNotFoundError(
            f"{config.csv_path} not found. Run: uv run python scripts/download_data.py"
        )
    return pd.read_csv(config.csv_path)


def clean(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Fix known problems and return a table with only feature columns + a 0/1 target."""
    df = df.copy()  # never change the caller's DataFrame

    # 1. Fail early and clearly if the CSV doesn't have the columns the config expects.
    expected = set(config.feature_columns) | {config.target}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"[{config.name}] columns missing from CSV: {sorted(missing)}")

    # 2. Text columns that should be numbers (Telco TotalCharges):
    #    to_numeric(errors="coerce") turns blanks into NaN, then fillna puts in the chosen value.
    for column, fill_value in config.fix_numeric.items():
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fill_value)

    # 3. Target as 0/1: 1 = churned, 0 = stayed. Telco "Yes"/"No" and Bank 1/0 both end up the same.
    df[config.target] = (df[config.target] == config.positive_label).astype(int)

    # 4. Keep only the columns the model may use, plus the target (this drops IDs and Surname).
    return df[config.feature_columns + [config.target]]


def split(
    df: pd.DataFrame, config: DatasetConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test sets.

    X = the input columns, y = the answer (churned or not).
    stratify=y keeps the same churn % in both sets, so the test set is a fair sample.
    """
    X = df.drop(columns=config.target)
    y = df[config.target]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def load_dataset(name: str):
    """One call does it all: load -> clean -> split. Returns X_train, X_test, y_train, y_test."""
    config = get_config(name)
    return split(clean(load_raw(config), config), config)
