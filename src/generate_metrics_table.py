#!/usr/bin/env python3
import csv
from pathlib import Path

# Define paths relative to the script location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
METRICS_PATH = PROJECT_ROOT / "results" / "metrics.csv"
README_PATH = PROJECT_ROOT / "README.md"


def format_metric(val):
    try:
        f_val = float(val)
        if f_val > 1.0:
            f_val = f_val / 100.0
        return f"{f_val:.4f}".replace(".", ",")
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


def generate_chart(rows):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Create DataFrame from rows
        df = pd.DataFrame(rows)
        # Convert numeric columns
        for col in ["accuracy", "precision", "recall", "f1"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Map exp names and classifier names for cleaner display
        exp_map = {
            "baseline": "Baseline",
            "experiment1": "Experiment 1",
            "experiment2": "Experiment 2",
            "experiment3": "Experiment 3",
            "experiment4": "Experiment 4",
            "experiment5": "Experiment 5",
            "experiment6": "Experiment 6",
            "experiment7": "Experiment 7",
            "experiment8": "Experiment 8",
        }
        clf_map = {
            "rf": "Random Forest",
            "svm": "SVM",
            "knn": "KNN",
        }
        df["experiment_label"] = df["experiment_name"].str.lower().map(exp_map).fillna(df["experiment_name"])
        df["classifier_label"] = df["classifier"].str.lower().map(clf_map).fillna(df["classifier"])

        # Sort values
        df["sort_key"] = df.apply(get_sort_key, axis=1)
        df = df.sort_values("sort_key")

        # Plot
        plt.figure(figsize=(11, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(
            data=df,
            x="experiment_label",
            y="accuracy",
            hue="classifier_label",
            palette="muted"
        )

        plt.title("Perbandingan Akurasi Model antar Eksperimen", fontsize=14, pad=15)
        plt.xlabel("Eksperimen", fontsize=12, labelpad=10)
        plt.ylabel("Akurasi", fontsize=12, labelpad=10)
        plt.ylim(0, 1.0)
        plt.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc='upper left')

        # Add value labels on top of bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=9, color='black',
                            xytext=(0, 3),
                            textcoords='offset points')

        fig_dir = PROJECT_ROOT / "results" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        chart_path = fig_dir / "metrics_comparison.png"

        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Comparison chart successfully saved to {chart_path}")
        return True
    except Exception as e:
        print(f"Warning: Failed to generate chart: {e}")
        return False


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

        # Generate comparison chart
        generate_chart(rows)

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
            "experiment5": "Experiment 5",
            "experiment6": "Experiment 6",
            "experiment7": "Experiment 7",
            "experiment8": "Experiment 8",
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
        # Embed the comparison chart image right under the table
        new_table_str += "\n\n### Grafik Perbandingan Akurasi\n\n![Comparison Chart](results/figures/metrics_comparison.png)"

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
