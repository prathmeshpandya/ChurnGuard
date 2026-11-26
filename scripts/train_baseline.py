"""Day 3: train the baseline models on both datasets and compare them on the test set.

    uv run python scripts/train_baseline.py

Saves:
    reports/day3_baseline.json                 all metrics
    reports/figures/day3_confusion_<name>.png  confusion matrices per dataset
"""

import json

import matplotlib

matplotlib.use("Agg")  # draw charts to files only, without opening windows
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

from churnguard.config import DATASETS, PROJECT_ROOT  # noqa: E402
from churnguard.data import load_dataset  # noqa: E402
from churnguard.evaluate import evaluate, plot_confusion_matrix  # noqa: E402
from churnguard.models import MODEL_KINDS, build_model  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, config in DATASETS.items():
        X_train, X_test, y_train, y_test = load_dataset(name)
        all_results[name] = {}

        # One row of charts per dataset: one confusion matrix per model.
        fig, axes = plt.subplots(1, len(MODEL_KINDS), figsize=(5 * len(MODEL_KINDS), 4))

        print(f"\n=== {name.upper()} (test set: {len(y_test)} customers, {y_test.sum()} churned) ===")
        print(f"{'model':<17}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>7}{'roc_auc':>9}")

        for kind, ax in zip(MODEL_KINDS, axes):
            model = build_model(config, kind)
            model.fit(X_train, y_train)            # learn from TRAIN only
            metrics = evaluate(model, X_test, y_test)  # score on unseen TEST
            all_results[name][kind] = metrics

            print(
                f"{kind:<17}{metrics['accuracy']:>9.3f}{metrics['precision']:>10.3f}"
                f"{metrics['recall']:>8.3f}{metrics['f1']:>7.3f}{metrics['roc_auc']:>9.3f}"
            )
            plot_confusion_matrix(model, X_test, y_test, f"{name}: {kind}", ax)

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"day3_confusion_{name}.png", dpi=120)
        plt.close(fig)

    with open(REPORTS_DIR / "day3_baseline.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved metrics and charts to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
