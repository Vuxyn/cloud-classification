#!/usr/bin/env python3
import csv
from pathlib import Path

# Define paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.csv"
README_PATH = PROJECT_ROOT / "README.md"


def format_metric(val):
    """Formats numeric metric values to percentages with two decimal places."""
    try:
        f_val = float(val)
        if 0.0 <= f_val <= 1.0:
            return f"{f_val * 100:.2f}%"
        return f"{f_val:.2f}%"
    except (ValueError, TypeError):
        return val


def get_sort_key(item):
    """Sort key to order by experiment sequence and then by classifier type."""
    exp_name = item.get("experiment_name", "").lower()
    clf = item.get("classifier", "").lower()

    if exp_name == "baseline":
        exp_idx = 0
    elif exp_name.startswith("experiment"):
        try:
            # Extract digits to support names like experiment1, experiment_2, etc.
            digits = "".join(filter(str.isdigit, exp_name))
            exp_idx = int(digits) if digits else 99
        except ValueError:
            exp_idx = 99
    else:
        exp_idx = 99

    clf_idx = {"rf": 0, "svm": 1, "knn": 2}.get(clf, 9)
    return (exp_idx, clf_idx)


def main():
    if not METRICS_PATH.exists():
        print(f"Error: {METRICS_PATH} not found.")
        return

    # Read CSV rows
    rows = []
    with open(METRICS_PATH, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment_name"):  # skip empty lines
                rows.append(row)

    if not rows:
        print("Metrics file is empty or contains no valid rows.")
        new_table_str = "*Belum ada hasil eksperimen.*"
    else:
        # Sort rows to ensure a consistent, professional order
        rows.sort(key=get_sort_key)

        table_rows = [
            "| Experiment | Classifier | Accuracy | Precision | Recall | F1-Score |",
            "|---|---|---|---|---|---|",
        ]

        exp_map = {
            "baseline": "Baseline",
            "experiment1": "Experiment 1",
            "experiment2": "Experiment 2",
            "experiment3": "Experiment 3",
            "experiment4": "Experiment 4",
        }

        clf_map = {
            "rf": "Random Forest",
            "svm": "SVM",
            "knn": "KNN",
        }

        for row in rows:
            exp_name = row.get("experiment_name", "")
            formatted_exp = exp_map.get(exp_name.lower(), exp_name.replace("_", " ").title())

            clf = row.get("classifier", "")
            formatted_clf = clf_map.get(clf.lower(), clf.upper())

            acc = format_metric(row.get("accuracy", "0"))
            prec = format_metric(row.get("precision", "0"))
            rec = format_metric(row.get("recall", "0"))
            f1 = format_metric(row.get("f1", "0"))

            table_rows.append(
                f"| {formatted_exp} | {formatted_clf} | {acc} | {prec} | {rec} | {f1} |"
            )

        new_table_str = "\n".join(table_rows)

    if README_PATH.exists():
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme_content = f.read()

        begin_tag = "<!-- BEGIN METRICS -->"
        end_tag = "<!-- END METRICS -->"

        if begin_tag in readme_content and end_tag in readme_content:
            parts = readme_content.split(begin_tag)
            before = parts[0]
            after_parts = parts[1].split(end_tag)
            after = after_parts[1]

            updated_content = f"{before}{begin_tag}\n{new_table_str}\n{end_tag}{after}"

            with open(README_PATH, "w", encoding="utf-8") as f:
                f.write(updated_content)
            print("README.md successfully updated with the metrics table.")
        else:
            print("Warning: Tags <!-- BEGIN METRICS --> and <!-- END METRICS --> not found in README.md.")
    else:
        print(f"Warning: {README_PATH} does not exist.")


if __name__ == "__main__":
    main()
