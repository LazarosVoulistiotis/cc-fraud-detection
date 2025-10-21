#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Χρήση:
  # BASH
python src/07_class_imbalance.py \
   --data data/data_raw/creditcard.csv \
   --target Class \
   --reports reports \
   --images reports/figures/week7 \
   --positive-label 1 \
   --normalize \
   --log-y \
   --title "Class Balance (Non-Fraud vs Fraud)" \
   --log-level INFO

  # CMD (Windows)
python src/07_class_imbalance.py --data data/data_raw/creditcard.csv --target Class --reports reports --images reports/figures/week7 --positive-label 1 --normalize --log-y --title "Class Balance (Non-Fraud vs Fraud)" --log-level INFO

Outputs:
- Bar plot (counts):   reports/figures/week7/class_balance_bar.png (ή .svg)
- Bar plot (percent):  reports/figures/week7/class_balance_pct.png (αν δώσεις --normalize)
- JSON σύνοψη:         reports/week7_class_balance/week7_class_balance.json
"""

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless export (CI/CD, servers)
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__) # module-level logger ώστε να ελέγχεις επίπεδο logs από το CLI (--log-level)


def ensure_dir(p: Path) -> None:
    """Δημιουργεί φάκελο αν δεν υπάρχει (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


def compute_class_balance(
    y: pd.Series,
    positive_label: int | str = 1 # Το int | str κάνει τη συνάρτηση ανθεκτική: δουλεύει είτε τα labels είναι 0/1 είτε '0'/'1'
) -> dict:
    """
    Επιστρέφει counts, percents, fraud_rate, imbalance metrics.
    Δεν υποθέτει ότι οι κλάσεις είναι μόνο 0/1 ούτε ότι η '1' είναι δεύτερη.
    """
    counts = y.value_counts().sort_index() # value_counts() δίνει πλήθος ανά κλάση, αγνοώντας NaN (αν υπήρχαν)
    total = int(counts.sum())

    percents = (counts / total * 100.0).to_dict() # Μετατρέπει τα counts σε ποσοστά % (όχι αναλογία). Το .to_dict() για εύκολο serialization
    
    # Μετατροπές για JSON serialization
    counts_d   = {str(k): int(v) for k, v in counts.items()}
    percents_d = {str(k): float(v) for k, v in percents.items()}
    # Fraud rate (ποσοστό της positive κλάσης)
    pos_key = str(positive_label)
    fraud_rate = float(percents_d.get(pos_key, 0.0))
    # Λόγος ανισορροπίας (majority/minority). Αν η μικρή κλάση έχει 0 δείγματα, δίνουμε None για να μην διαιρέσουμε με μηδέν
    maj = counts.max()
    min_ = counts.min()
    major_to_minor_ratio = float(maj / min_) if min_ > 0 else None
    # Μεταφράζει το % σε διαισθητικό KPI: “~1 απάτη ανά Ν συναλλαγές”. Παράδειγμα στο κλασικό dataset: ~0.173% ⇒ ~1/579
    one_positive_per_n = (100.0 / fraud_rate) if fraud_rate > 0 else None
    # Labels και values για πλοτ
    labels = [str(k) for k in counts.index.tolist()]
    values = [int(v) for v in counts.values.tolist()]
    # Επιστρέφει όλα τα στοιχεία
    return {
        "labels": labels,
        "counts": counts_d,
        "percents": percents_d,
        "total": total,
        "fraud_rate": fraud_rate,
        "major_to_minor_ratio": major_to_minor_ratio,
        "one_positive_per_n": one_positive_per_n,
        "positive_label": str(positive_label),
        "values_for_plot": values
    }


def barplot(
    labels: list[str],
    heights: list[float],
    out_png: Path, 
    title: str,
    ylabel: str,
    positive_label: str = "1", # ποια κλάση είναι η positive (για χρωματισμό)
    annotate_percents: dict[str, float] | None = None, # αν δίνεται, προσθέτει % στα annotations
    log_y: bool = False
) -> None:
    """
    Δημιουργεί bar plot (counts ή %).
    - Χρωματίζει την positive κλάση διαφορετικά (business-friendly).
    - Προσθέτει annotations (count ή %).
    - Προαιρετικό log-scale στον άξονα y.
    - Numeric x + custom ticks για να αποφύγουμε τα matplotlib 'categorical units' logs.
    """
    ensure_dir(out_png.parent) # βεβαιώνεται ότι υπάρχει ο φάκελος εξόδου

    # Χρώματα (highlight το positive)
    colors = ["#5B8FF9" if lbl != positive_label else "#F4664A" for lbl in labels]

    # Numeric x-axis για να σιωπήσουν τα categorical logs
    x = np.arange(len(labels)) 
    plt.figure()
    bars = plt.bar(x, heights, color=colors)
    plt.title(title)
    plt.xlabel("Class label")
    plt.ylabel(ylabel)
    plt.xticks(x, labels)

    if log_y:
        plt.yscale("log") # log scale στον άξονα y

    # Annotations
    for bar, lbl, h in zip(bars, labels, heights):
        if ylabel.lower().startswith("count"):
            txt = f"{int(h)}"
            if annotate_percents is not None:
                p = annotate_percents.get(lbl, None)
                if p is not None:
                    txt = f"{int(h)}\n({p:.3f}%)"
        else:
            txt = f"{h:.3f}%"

        plt.text(bar.get_x() + bar.get_width()/2.0,
                 bar.get_height(),
                 txt,
                 ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()
    logger.debug(f"Saved plot: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str, help="CSV dataset path")
    ap.add_argument("--target", default="Class", type=str, help="Target column")
    ap.add_argument("--reports", default="reports", type=str, help="Base folder for reports")
    ap.add_argument("--images", default="reports/figures/week7", type=str, help="Folder for plots")
    # Flags
    ap.add_argument("--positive-label", default="1", type=str, help="Positive class label (π.χ. '1')")
    ap.add_argument("--normalize", action="store_true", help="Εξαγωγή plot και σε %")
    ap.add_argument("--log-y", action="store_true", help="Log κλίμακα στον άξονα y")
    ap.add_argument("--title", default="Class Balance (Non-Fraud vs Fraud)", type=str, help="Plot title")
    ap.add_argument("--images-format", default="png", choices=["png", "svg"])
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    args = ap.parse_args()

    # Logging(
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s") # βασική ρύθμιση logging (--log-level)
    # Σίγασε τα info logs της Matplotlib για "categorical units"
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING) # Σβήνει τα info logs της matplotlib για “categorical units”
    
    data_path   = Path(args.data)
    reports_dir = Path(args.reports)
    images_dir  = Path(args.images)
    ensure_dir(reports_dir)
    ensure_dir(images_dir)
    # Φόρτωση δεδομένων
    logger.info(f"Φόρτωση: {data_path}")
    df = pd.read_csv(data_path)
    assert args.target in df.columns, f"Target column '{args.target}' not found" # έλεγχος ύπαρξης target

    # Target → numeric (ρίχνουμε NAs μόνο για την οπτικοποίηση/report)
    y = pd.to_numeric(df[args.target], errors="coerce") # μετατρέπει σε numeric, βάζει NaN όπου αποτυγχάνει
    if y.isna().any():
        dropped = int(y.isna().sum())
        logger.warning(f"Target είχε {dropped} NaNs — αγνοούνται για την οπτικοποίηση.")
        y = y.dropna()
    if (y % 1 == 0).all():
        y = y.astype(int)

    # Υπολογισμοί
    summary = compute_class_balance(y, positive_label=args.positive_label)

    # Πλοτ 1: Counts
    out_counts = images_dir / f"class_balance_bar.{args.images_format}"
    barplot(
        labels=summary["labels"],
        heights=summary["values_for_plot"],
        out_png=out_counts,
        title=args.title,
        ylabel="Count",
        positive_label=summary["positive_label"],
        annotate_percents=summary["percents"],
        log_y=args.log_y
    )

    # Πλοτ 2: Percentages (προαιρετικό)
    if args.normalize:
        pct_heights = [summary["percents"][lbl] for lbl in summary["labels"]]
        out_pct = images_dir / f"class_balance_pct.{args.images_format}"
        barplot(
            labels=summary["labels"],
            heights=pct_heights,
            out_png=out_pct,
            title=args.title + " (Percent)",
            ylabel="Percent",
            positive_label=summary["positive_label"],
            annotate_percents=None,
            log_y=False
        )

    # JSON: στο reports/week7_class_balance/week7_class_balance.json
    out_json_dir = reports_dir / "week7_class_balance"
    ensure_dir(out_json_dir)
    out_json = out_json_dir / "week7_class_balance.json"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "target": args.target,
            "summary": summary,
            "business_notes": {
                "imbalance": "Η θετική κλάση είναι σπάνια — προτείνεται PR-AUC, Recall@k, cost-sensitive metrics, stratified split, class weights/SMOTE και thresholding με βάση κόστος.",
                "one_positive_per_n": summary.get("one_positive_per_n")
            }
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved plot (counts): {out_counts}")
    if args.normalize:
        logger.info(f"Saved plot (percent): {out_pct}")
    logger.info(f"Saved json: {out_json}")


if __name__ == "__main__":
    main()
