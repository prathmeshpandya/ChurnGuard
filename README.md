# ChurnGuard

Predict which customers are likely to leave (churn), using one reusable ML pipeline that works on two different datasets:

| Dataset | Customers | Target | Churn rate |
|---|---|---|---|
| Telco (IBM) | 7,043 | `Churn` (Yes/No) | 26.5% |
| Bank | 10,000 | `Exited` (1/0) | 20.4% |

**Stack:** Python 3.12, uv, pandas, matplotlib, seaborn, Jupyter. Coming later: scikit-learn, XGBoost, SHAP, FastAPI, Docker.

## Progress
- [x] **Day 1:** Project setup, data download, exploratory data analysis
- [ ] Day 2: Cleaning and preprocessing pipeline
- [ ] Day 3: Baseline model and metrics
- [ ] Day 4: Better models, cross-validation, tuning
- [ ] Day 5: Explaining predictions, saving models
- [ ] Day 6: FastAPI prediction service
- [ ] Day 7: Tests, Docker, deployment

## Run it
```bash
uv sync                                   # create .venv and install packages
uv run python scripts/download_data.py    # only if data/raw/ is empty
```
Then open `notebooks/01_eda.ipynb` in VS Code and pick the `.venv` Python kernel.

## Project structure
```
data/raw/            the two CSV datasets
notebooks/           exploration notebooks
scripts/             one-off commands (data download)
src/churnguard/      reusable pipeline code (from Day 2)
pyproject.toml       project info and package list
uv.lock              exact package versions
```

## Data sources
- Telco: [IBM telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Bank: "Churn Modelling" dataset, originally from Kaggle (copy downloaded from Hugging Face)
