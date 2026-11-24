"""Download the two churn datasets into data/raw/.

Run from the project folder:
    uv run python scripts/download_data.py
"""

from pathlib import Path
from urllib.request import urlretrieve

# Where the files go. Path(__file__) is this script; .parent.parent is the project folder.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# file name -> where to download it from
DATASETS = {
    # IBM's telecom churn dataset (7,043 customers), from IBM's official GitHub repo.
    "telco_churn.csv": (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    ),
    # Bank churn dataset (10,000 customers). Originally on Kaggle ("Churn_Modelling.csv");
    # this is a public copy on Hugging Face, because Kaggle needs a login.
    "bank_churn.csv": (
        "https://huggingface.co/spaces/KaustavModak/ann-classification-customer-churn/"
        "resolve/main/Churn_Modelling.csv"
    ),
}


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)  # create data/raw/ if it doesn't exist

    for filename, url in DATASETS.items():
        target = RAW_DIR / filename
        if target.exists():
            print(f"Skip {filename}: already downloaded")
            continue
        print(f"Downloading {filename} ...")
        urlretrieve(url, target)  # fetch the URL and save it to the target file
        size_kb = target.stat().st_size / 1024
        print(f"  saved to {target} ({size_kb:.0f} KB)")


# Only run main() when the file is run directly, not when it is imported.
if __name__ == "__main__":
    main()
