#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Το shebang επιτρέπει σε Unix-like συστήματα να τρέξεις το αρχείο ως εκτελέσιμο (./script.py) βρίσκοντας τον κατάλληλο interpreter από το περιβάλλον.
# Η δήλωση UTF-8 διασφαλίζει ότι τα ελληνικά στο αρχείο (σχόλια/strings) διαβάζονται σωστά. Στην Python 3 είναι ήδη default UTF-8, αλλά δεν βλάπτει.
"""
src/04_eda_time_amount.py

EDA για Time & Amount στο ULB creditcard.csv

Παράγει:
- <reports>/week5_eda_time_amount/eda_time_amount_stats.csv
- <reports>/week5_eda_time_amount/eda_time_amount_summary.json
- <reports>/05_eda_time_amount.md
- εικόνες στο <images> (histograms/densities)

Usage (EN): (PowerShell)
  python .\src\06_eda_time_amount.py `
  --data .\data\data_raw\creditcard.csv `
  --reports .\reports `
  --images .\reports\figures\week5 `
  --fig-format png `
  --log-level INFO `
  --show
or (CMD):
    python .\src\06_eda_time_amount.py --data .\data\data_raw\creditcard.csv --reports .\reports --images .\reports\figures\week5 --fig-format png --log-level INFO --show
or (Bash):
    python ./src/06_eda_time_amount.py --data ./data/data_raw/creditcard.csv --reports ./reports --images ./reports/figures/week5 --fig-format png --log-level INFO --show

Χωρίς GUI, άκουσε: μην δώσεις --show (ή τρέξε με backend Agg).
"""

from __future__ import annotations # Κάνει defer την αξιολόγηση των type annotations σε strings (postponed evaluation).
import argparse      # παίρνεις --data, --reports, --images, κ.λπ. από τη γραμμή εντολών
import json          # γράφεις machine-readable σύνοψη (χρήσιμο σε CI/automations)
import logging       # κονσόλα με timestamps/levels αντί για raw print
from pathlib import Path   # cross-platform paths (Windows/Unix)
from typing import Dict, Any  # type hints για συναρτήσεις

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend by default
import matplotlib.pyplot as plt


# --------------------------- Utilities ---------------------------

def setup_logging(level: str = "INFO") -> None:
    """Basic logging setup."""
    numeric = getattr(logging, level.upper(), logging.INFO) # Μετατρέπει το string (π.χ. "INFO", "DEBUG", "WARNING") στο αντίστοιχο numeric level του logging (π.χ. logging.INFO = 20). Αν δώσεις κάτι άκυρο, κάνει default σε INFO.
    logging.basicConfig( # Ρυθμίζει το βασικό configuration του logging.
        level=numeric, # Επίπεδο logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s [%(levelname)s] %(message)s", # θα εμφανίζεται ώρα, επίπεδο και κείμενο
        datefmt="%H:%M:%S", # ώρα:λεπτό:δευτερόλεπτο (όχι ημερομηνία)
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True) # parents=True: φτιάχνει και τους ενδιάμεσους φακέλους αν δεν υπάρχουν
    # Χρήση: πριν από savefig, to_csv, open(...) κ.λπ., καλείς ensure_dir(Path_dir) για να μη “σκάει” από missing directory

def qualit_skew(series: pd.Series) -> str: # να γυρίζει λεκτική περιγραφή της ασυμμετρίας (skewness) μιας στήλης για το report
    """Qualitative description of skewness."""
    s = float(series.skew()) # υπολογίζει το sample skewness (Pearson’s moment coefficient)
    if s > 2:
        return "πολύ έντονα δεξιά ασύμμετρη (heavy right-skew)"
    if s > 1:
        return "έντονα δεξιά ασύμμετρη"
    if s < -1:
        return "έντονα αριστερά ασύμμετρη"
    return "σχετικά συμμετρική/ήπια ασύμμετρη"
    # μπαίνει αυτούσιο στο Markdown report ώστε να μην γράφεις χειροκίνητα ποιοτική περιγραφή

def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    """Render DataFrame as GitHub-friendly markdown table."""
    # round numerical columns for readability
    out = df.copy() # για να μη αλλάξει το αρχικό df
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda x: f"{x:{floatfmt}}" if pd.notna(x) else "") # f"{x:{floatfmt}}" με default ".6g" → έως 6 σημαντικά ψηφία, αυτόματα επιστημονική αν χρειάζεται, else empty string for NaN
    return out.to_markdown(index=True) # πίνακα Markdown με index στην αριστερή στήλη

# Σκοπός: αποθήκευση δομημένων αποτελεσμάτων (π.χ. σύνοψη EDA) σε JSON για downstream κατανάλωση (π.χ. scripts, dashboards).
def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    # indent=2 για όμορφο format, ensure_ascii=False κρατά ελληνικά/Unicode ως έχουν (όχι \u03bb escapes)


# --------------------------- Plot helpers ---------------------------

def plot_and_save(
    fig_path: Path, # path χωρίς κατάληξη (θα προστεθεί από fmt)
    fig_maker, # συνάρτηση που παίρνει ax και φτιάχνει το γράφημα
    *, # υποχρεωτικά ονομαστικά arguments μετά το *
    dpi: int = 150, # ανάλυση εικόνας
    bbox: str = "tight", # κόβει τα περιθώρια γύρω από το γράφημα
    show: bool = False, # αν True, καλεί plt.show() (αν έχει interactive backend)
    fmt: str = "png" # μορφή εικόνας (png, jpg, svg, pdf)
):
    """Create a figure using `fig_maker(ax)` and save it."""
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        fig_maker(ax)
        fig.tight_layout()
        fig_path = fig_path.with_suffix("." + fmt) # αλλάζει extension (αν έδωσες “eda_plot” → γίνεται “eda_plot.png”
        fig.savefig(fig_path, dpi=dpi, bbox_inches=bbox)
        logging.info("Saved figure → %s", fig_path.as_posix())
        if show:
            # If user passed --show and has an interactive backend
            try:
                plt.show()
            except Exception as e:
                logging.warning("Non-interactive backend; cannot show: %s", e)
    finally:
        plt.close(fig) # Κλείνει τη φιγούρα για να μη γεμίζει η μνήμη (ειδικά σε loops)
    """     Γιατί είναι χρήσιμο
            - Σου επιτρέπει να γράφεις μία γραμμή κώδικα για κάθε plot αντί για boilerplate.
            - Το μόνο που αλλάζει κάθε φορά είναι η fig_maker(ax).
            - Ελέγχει consistent parameters (dpi, format, layout) για όλα τα plots.
            - Σου κάνει logging σε κάθε αποθήκευση → ευκολότερο debugging."""
    """ Παράδειγμα χρήσης:  
        plot_and_save(
        Path("reports/figures/week5/eda_amount_hist"),
        lambda ax: ax.hist(df["Amount"], bins=100),
        show=True
      )
    """

# --------------------------- Core EDA ---------------------------

def run_eda(
    data_path: Path,
    reports_dir: Path,
    images_dir: Path,
    fig_format: str = "png",
    show: bool = False,
) -> None:
    # Load
    if not data_path.exists():
        raise FileNotFoundError(f"Λείπει το {data_path}")
    ensure_dir(reports_dir)
    ensure_dir(images_dir)

    logging.info("Loading dataset: %s", data_path.as_posix())
    df = pd.read_csv(data_path, low_memory=False) # low_memory=False κάνει πιο αξιόπιστη την αναγνώριση τύπων (αποφεύγει mixed dtypes)

    if "Class" not in df.columns or "Amount" not in df.columns or "Time" not in df.columns:
        raise ValueError("Αναμένω στήλες 'Class', 'Amount', 'Time' στο dataset.")


    # ---------- Descriptive stats ----------
    num_cols = ["Time", "Amount"]
    desc = df[num_cols].describe( # περιλαμβάνει count, mean, std, min, 25%, 50%, 75%, max
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99] # πρόσθετα percentiles
    ).T # transpose ώστε οι γραμμές να είναι οι στήλες (Time, Amount) και οι στήλες τα metrics
    desc["skew"] = df[num_cols].skew() 
    desc["kurtosis"] = df[num_cols].kurtosis() 

    # Save CSV + Markdown
    stats_csv = (reports_dir / "week5_eda_time_amount/eda_time_amount_stats.csv")
    desc.to_csv(stats_csv, index=True) # index=True για να γράψει και το όνομα της στήλης (Time, Amount)
    logging.info("Saved stats CSV → %s", stats_csv.as_posix()) 

    stats_md = (reports_dir / "week5_eda_time_amount/eda_time_amount_stats.md")
    md_table = df_to_markdown(desc, floatfmt=".6g") 
    with open(stats_md, "w", encoding="utf-8") as f:
        f.write("# Περιγραφικά στατιστικά (Time, Amount)\n\n")
        f.write(md_table + "\n")
    logging.info("Saved stats Markdown → %s", stats_md.as_posix())


    # ---------- Plots ----------
    # Time (seconds)
    plot_and_save(
        images_dir / "eda_time_hist_seconds",
        lambda ax: ( # tuple για να μπορείς να κάνεις πολλαπλές εντολές μέσα στο lambda
            ax.hist(df["Time"].values, bins=100, density=False), # histogram με 100 bins
            ax.set_title("Histogram of Time (seconds from first transaction)"),
            ax.set_xlabel("Time (seconds)"),
            ax.set_ylabel("Count"),
        ),
        show=show, fmt=fig_format
    )

    # Time → hour-of-day
    hours = ((df["Time"] % 86400) // 3600).astype(int) # 86400 = 24*3600 seconds in a day, 3600 seconds in an hour
    plot_and_save(
        images_dir / "eda_time_hist_hour",
        lambda ax: (
            ax.hist(hours, bins=np.arange(-0.5, 24.5, 1.0), rwidth=0.9),
            ax.set_title("Histogram of Hour-of-Day derived from Time"),
            ax.set_xlabel("Hour of Day [0-23]"),
            ax.set_ylabel("Count"),
        ),
        show=show, fmt=fig_format
    )

    # Amount (linear)
    plot_and_save(
        images_dir / "eda_amount_hist",
        lambda ax: (
            ax.hist(df["Amount"].values, bins=100, density=False), # density=False για counts, όχι probability density
            ax.set_title("Histogram of Amount"),
            ax.set_xlabel("Amount"),
            ax.set_ylabel("Count"),
        ),
        show=show, fmt=fig_format
    )

    # Amount (log x-scale)
    plot_and_save(
        images_dir / "eda_amount_hist_logx",
        lambda ax: (
            ax.hist(df["Amount"].values, bins=100, density=False),
            ax.set_xscale("log"), # Όταν τα ποσά έχουν μεγάλη ασυμμετρία (πολλά μικρά ποσά, λίγα πολύ μεγάλα), ο λογάριθμος “στρώνει” το γράφημα
            ax.set_title("Histogram of Amount (log x-scale)"),
            ax.set_xlabel("Amount (log scale)"),
            ax.set_ylabel("Count"),
        ),
        show=show, fmt=fig_format
    )

    # Amount log1p
    amount_log1p = np.log1p(df["Amount"].clip(lower=0)) # log1p(x) = log(1+x), clip για να αποφύγουμε αρνητικά/NaN (αν και δεν περιμένουμε)
    plot_and_save(
        images_dir / "eda_amount_hist_log1p",
        lambda ax: (
            ax.hist(amount_log1p.values, bins=100, density=False),
            ax.set_title("Histogram of log1p(Amount)"),
            ax.set_xlabel("log1p(Amount)"),
            ax.set_ylabel("Count"),
        ),
        show=show, fmt=fig_format
    )

    # By Class densities
    fraud = df[df["Class"] == 1]
    nonfraud = df[df["Class"] == 0]

    # Amount vs Class (overlapping histograms)
    plot_and_save(
        images_dir / "eda_amount_by_class_density",
        lambda ax: (
            ax.hist(nonfraud["Amount"].values, bins=80, density=True, alpha=0.5, label="Class 0 (non-fraud)"),
            ax.hist(fraud["Amount"].values,    bins=80, density=True, alpha=0.5, label="Class 1 (fraud)"),
            ax.set_title("Amount distribution by Class (density)"),
            ax.set_xlabel("Amount"),
            ax.set_ylabel("Density"),
            ax.legend(),
        ),
        show=show, fmt=fig_format
    )

    # log1p(Amount) by Class (overlapping histograms)
    plot_and_save(
        images_dir / "eda_amount_log1p_by_class_density",
        lambda ax: (
            ax.hist(np.log1p(nonfraud["Amount"].clip(lower=0)), bins=80, density=True, alpha=0.5, label="Class 0 (non-fraud)"),
            ax.hist(np.log1p(fraud["Amount"].clip(lower=0)),    bins=80, density=True, alpha=0.5, label="Class 1 (fraud)"),
            ax.set_title("log1p(Amount) by Class (density)"),
            ax.set_xlabel("log1p(Amount)"),
            ax.set_ylabel("Density"),
            ax.legend(),
        ),
        show=show, fmt=fig_format
    )

    # Hour-of-day by Class (overlapping histograms)
    fraud_hours = ((fraud["Time"] % 86400) // 3600).astype(int)
    nonfraud_hours = ((nonfraud["Time"] % 86400) // 3600).astype(int)
    plot_and_save(
        images_dir / "eda_hour_by_class_density",
        lambda ax: (
            ax.hist(nonfraud_hours, bins=np.arange(-0.5, 24.5, 1.0), density=True, alpha=0.5, label="Class 0 (non-fraud)", rwidth=0.9),
            ax.hist(fraud_hours,    bins=np.arange(-0.5, 24.5, 1.0), density=True, alpha=0.5, label="Class 1 (fraud)", rwidth=0.9),
            ax.set_title("Hour-of-Day distribution by Class (density)"),
            ax.set_xlabel("Hour of Day [0-23]"),
            ax.set_ylabel("Density"),
            ax.legend(),
        ),
        show=show, fmt=fig_format
    )


    # ---------- Findings / Summary (Markdown + JSON) ----------
    amount_skew_txt = qualit_skew(df["Amount"]) # Καλεί τη helper qualit_skew και επιστρέφει φιλική περιγραφή (π.χ. “έντονα δεξιά ασύμμετρη”)
    time_skew_txt   = qualit_skew(df["Time"])
    q_amt_99        = float(df["Amount"].quantile(0.99)) # 99ο εκατοστημόριο (cutoff για “top 1%” συναλλαγών)
    pct_amt_big     = float((df["Amount"] > q_amt_99).mean() * 100.0) # Ποιο ποσοστό των συναλλαγών είναι πάνω από αυτό το όριο
    fraud_rate      = float(df["Class"].mean()) # Συνολικό fraud rate (μέσος όρος της δυαδικής Class).

    hour_all = ((df["Time"] % 86400) // 3600).astype(int) # Ώρα της ημέρας [0-23] από το Time
    hour_tbl = pd.DataFrame({"hour": hour_all, "is_fraud": df["Class"].astype(int)}) # DataFrame με δύο στήλες: hour (0-23) και is_fraud (0/1)
    agg = hour_tbl.groupby("hour").agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum")) # Ομαδοποίηση κατά hour, μετράει πόσες συναλλαγές (total) και πόσες απάτες (fraud) ανά ώρα
    agg["fraud_rate_hour"] = agg["fraud"] / agg["total"] # Υπολογίζει πλήθος/αριθμό fraud ανά ώρα και φτιάχνει fraud_rate_hour
    top_hours = agg.sort_values("fraud_rate_hour", ascending=False).head(3) # Κρατά τις top-3 ώρες με το υψηλότερο fraud rate (ενδεικτικό pattern)

    top_hours_list = ", ".join(
        [f"{int(h)} (rate={r:.3f})" for h, r in zip(top_hours.index, top_hours["fraud_rate_hour"])]
    )

    md = f"""---
# Εβδομάδα 5 — Ευρήματα EDA (Time & Amount)

- **Fraud rate (συνολικό):** {fraud_rate:.4%}
- **Κατανομή Amount:** {amount_skew_txt}. Το άνω 1% των ποσών ξεκινά περίπου από **{q_amt_99:.2f}** (περίπου {pct_amt_big:.2f}% των συναλλαγών).
- **Κατανομή Time:** {time_skew_txt}. Θυμήσου ότι το `Time` είναι δευτερόλεπτα από την πρώτη συναλλαγή.
- **Ώρες με αυξημένο ποσοστό απάτης (ενδεικτικά):** {top_hours_list}
- **Οπτικοποιήσεις:** δες εικόνες στον φάκελο `{images_dir.as_posix()}`:
  - `eda_time_hist_seconds.{fig_format}`, `eda_time_hist_hour.{fig_format}`
  - `eda_amount_hist.{fig_format}`, `eda_amount_hist_logx.{fig_format}`, `eda_amount_hist_log1p.{fig_format}`
  - `eda_amount_by_class_density.{fig_format}`, `eda_amount_log1p_by_class_density.{fig_format}`, `eda_hour_by_class_density.{fig_format}`

## Παρατηρήσεις
1. Η κατανομή του `Amount` είναι συχνά δεξιά ασύμμετρη — η **log1p** μετασχημάτιση βελτιώνει την ορατότητα.
2. Οι κατανομές ανά κλάση δείχνουν αν οι **απάτες** συμβαίνουν σε **διαφορετικά εύρη ποσών** ή **συγκεκριμένες ώρες**.
3. Τα patterns στις ώρες είναι **ενδεικτικά** και **όχι αιτιώδη** — απαιτούν περαιτέρω διερεύνηση.
---
"""
    md_path = reports_dir / "05_eda_time_amount.md" # Ο τελεστής / στο Path κάνει ασφαλές join των κομματιών του path (αντί για string concatenation)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    logging.info("Saved summary Markdown → %s", md_path.as_posix()) # as_posix() γυρνά το path σε POSIX format με / (π.χ. χρήσιμο για logs/CI σε διαφορετικά λειτουργικά)

    # JSON summary (για αυτοματοποίηση/CI)
    summary = { # Φτιάχνει ένα λεξικό (dict) με συνοπτικά ευρήματα και μετα-δεδομένα (artifacts/paths).
        "rows": int(len(df)),
        "fraud_rate": fraud_rate,
        "amount": {
            "skew_desc": amount_skew_txt,
            "p99": q_amt_99,
            "pct_above_p99": pct_amt_big / 100.0,
        },
        "time": {"skew_desc": time_skew_txt},
        "top_hours_by_fraud_rate": [
            {"hour": int(h), "rate": float(r)} for h, r in zip(top_hours.index, top_hours["fraud_rate_hour"])
        ],
        "artifacts": {
            "stats_csv": stats_csv.as_posix(),
            "stats_md": stats_md.as_posix(),
            "summary_md": md_path.as_posix(),
            "figures_dir": images_dir.as_posix(),
        },
    }
    json_path = reports_dir / "week5_eda_time_amount/eda_time_amount_summary.json"
    save_json(summary, json_path)
    logging.info("Saved JSON summary → %s", json_path.as_posix())

    logging.info("Done.")


# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "EDA για Time & Amount στο creditcard.csv "
            "(παράγει στατιστικά, εικόνες και markdown/json αναφορά)."
        )
    )
    p.add_argument("--data", required=True, type=str,
                   help="Διαδρομή προς το creditcard.csv (e.g. data/data_raw/creditcard.csv)")
    p.add_argument("--reports", default="reports", type=str,
                   help="Φάκελος αναφορών/εξόδων (default: reports)")
    p.add_argument("--images", default=None, type=str,
                   help="Φάκελος εικόνων (default: <reports>/figures)")
    p.add_argument("--fig-format", default="png", choices=["png", "jpg", "svg", "pdf"],
                   help="Μορφή εικόνων που θα εξαχθούν (default: png)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Επίπεδο logging (default: INFO)")
    p.add_argument("--show", action="store_true",
                   help="Προβολή γραφημάτων (απαιτεί interactive backend).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    data_path = Path(args.data)
    reports_dir = Path(args.reports)
    images_dir = Path(args.images) if args.images else (reports_dir / "figures")

    run_eda(
        data_path=data_path,
        reports_dir=reports_dir,
        images_dir=images_dir,
        fig_format=args.fig_format,
        show=args.show,
    )


if __name__ == "__main__":
    main()
