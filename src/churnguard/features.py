"""Turn raw columns into numbers a model can learn from.

Models only understand numbers. This builds a scikit-learn "preprocessor" that:
  - numeric columns:     fill any gaps with the median, then scale to a similar range
  - categorical columns: fill any gaps with the most common value, then one-hot encode
It is built here but *fitted* (taught) on training data only, to avoid data leakage.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnguard.config import DatasetConfig


def build_preprocessor(config: DatasetConfig) -> ColumnTransformer:
    # A Pipeline runs its steps in order: the output of one step is the input to the next.
    numeric_steps = Pipeline([
        # Our cleaned data has no gaps, but the API (Day 6) might receive some.
        ("impute", SimpleImputer(strategy="median")),
        # Rescale so each column has mean 0 and spread 1. Otherwise Balance (~100,000)
        # would dwarf Age (~40) for models like logistic regression.
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # One-hot: "Contract" becomes Contract_Month-to-month / Contract_One year / Contract_Two year,
        # each 0 or 1. handle_unknown="ignore": a category never seen in training becomes all zeros
        # instead of crashing.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # ColumnTransformer sends each group of columns through its own pipeline, then glues the
    # results side by side into one table of numbers.
    return ColumnTransformer([
        ("num", numeric_steps, config.numeric_columns),
        ("cat", categorical_steps, config.categorical_columns),
    ])
