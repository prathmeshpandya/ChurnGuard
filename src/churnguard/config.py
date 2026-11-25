"""Settings for each churn dataset.

The pipeline code (data.py, features.py) never mentions "Telco" or "Bank" directly.
Everything dataset-specific lives here, so adding a third dataset = adding one more config.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project folder = three levels up from this file (src/churnguard/config.py -> ChurnGuard/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Same random seed everywhere, so splits and models are reproducible (same result every run).
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% of customers are kept aside for testing


# frozen=True: once created, a config can't be changed by accident.
@dataclass(frozen=True)
class DatasetConfig:
    name: str                       # short id, e.g. "telco"
    csv_file: str                   # file name inside data/raw/
    target: str                     # the column we want to predict
    positive_label: str | int       # the value in `target` that means "churned"
    numeric_columns: list[str]      # columns with numbers
    categorical_columns: list[str]  # columns with categories (text labels)
    drop_columns: list[str] = field(default_factory=list)  # IDs etc., never used by the model
    # Columns stored as text that should be numbers, with the value to use when a row is blank.
    fix_numeric: dict[str, float] = field(default_factory=dict)

    @property
    def csv_path(self) -> Path:
        return RAW_DATA_DIR / self.csv_file

    @property
    def feature_columns(self) -> list[str]:
        """All columns the model is allowed to see."""
        return self.numeric_columns + self.categorical_columns


TELCO = DatasetConfig(
    name="telco",
    csv_file="telco_churn.csv",
    target="Churn",
    positive_label="Yes",
    numeric_columns=["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"],
    categorical_columns=[
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ],
    drop_columns=["customerID"],
    # Day 1 finding: 11 new customers (tenure 0) have a blank TotalCharges -> they owe 0 so far.
    fix_numeric={"TotalCharges": 0.0},
)

BANK = DatasetConfig(
    name="bank",
    csv_file="bank_churn.csv",
    target="Exited",
    positive_label=1,
    numeric_columns=[
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ],
    categorical_columns=["Geography", "Gender"],
    # IDs carry no pattern; Surname could make the model unfair.
    drop_columns=["RowNumber", "CustomerId", "Surname"],
)

# Look up a config by name: DATASETS["telco"]
DATASETS: dict[str, DatasetConfig] = {cfg.name: cfg for cfg in (TELCO, BANK)}


def get_config(name: str) -> DatasetConfig:
    """Return the config for a dataset name, with a clear error for unknown names."""
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}. Choose one of: {list(DATASETS)}") from None
