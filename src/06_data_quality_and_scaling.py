#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Χρήση:

-BASH-
python src/06_data_quality_and_scaling.py \
  --data data/data_raw/creditcard.csv \
  --reports reports/week6_data_quality_and_amount_scaling \
  --images reports/figures/week6 \
  --out data/data_interim/creditcard_amount_scaled.csv \
  --method auto \
  --require-cols Amount,Class \
  --max-nulls 0 \
  --max-duplicates 0 \
  --log-level INFO \
  --drop-duplicates

-CMD-
python src/06_data_quality_and_scaling.py --data data/data_raw/creditcard.csv --reports reports/week6_data_quality_and_amount_scaling --images reports/figures/week6 --out data/data_interim/creditcard_amount_scaled.csv --method auto --require-cols Amount,Class --max-nulls 0 --max-duplicates 0 --log-level INFO --drop-duplicates

Σκοπός:
- Έλεγχος πληρότητας & ποιότητας (nulls, duplicates, αρνητικές τιμές, περιγραφικά)
- Αυτόματη κλιμάκωση Amount (z-score ή log1p+z-score) βάσει skewness
- Εξαγωγή αναφοράς JSON + γραφημάτων πριν/μετά

ΣΗΜ.:
- Η κλιμάκωση εδώ είναι για EDA/τεκμηρίωση. Στην εκπαίδευση θα γίνει ΜΟΝΟ σε X_train μέσω
  scikit-learn Pipeline (για αποφυγή data leakage).
- Για περιγραφικά (reporting) χρησιμοποιούμε sample std (ddof=1).
  Για τον ίδιο τον μετασχηματισμό (standardize) χρησιμοποιούμε population std (ddof=0).
"""

from __future__ import annotations
from typing import TypedDict
from pathlib import Path
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# --------------------------
# Βοηθητικές συναρτήσεις
# --------------------------

def ensure_dir(p: Path) -> None:
    """Δημιουργεί φάκελο p (και γονείς) αν δεν υπάρχει."""
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Αδυναμία δημιουργίας φακέλου: {p} ({e})")
        raise

class AmountStats(TypedDict, total=False):
    count: int
    mean: float
    std: float
    min: float
    max: float
    skew: float
    kurtosis: float
    q01: float; q05: float; q25: float; q50: float; q75: float; q95: float; q99: float
    non_negative_ratio: float
    empty: bool

def describe_amount(a: pd.Series) -> AmountStats:
    """Περιγραφικά για Amount με καθάρισμα NaN/Inf (sample std, ddof=1)."""
    a = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if a.empty:
        return AmountStats(empty=True)
    q = a.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
    return AmountStats(
        count=int(a.count()),
        mean=float(a.mean()),
        std=float(a.std(ddof=1)),
        min=float(a.min()),
        max=float(a.max()),
        skew=float(a.skew()),
        kurtosis=float(a.kurtosis()),
        q01=float(q[0.01]), q05=float(q[0.05]), q25=float(q[0.25]), q50=float(q[0.5]),
        q75=float(q[0.75]), q95=float(q[0.95]), q99=float(q[0.99]),
        non_negative_ratio=float((a >= 0).mean()),
        empty=False
    )

def plot_hist(series: pd.Series, title: str, outpath: Path, bins: int | str = "auto") -> Path:
    """Αποθηκεύει ιστόγραμμα και επιστρέφει το path."""
    ensure_dir(outpath.parent)
    plt.figure()
    plt.hist(series, bins=bins)
    plt.title(title); plt.xlabel("Value"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(outpath, dpi=140); plt.close()
    logger.debug(f"Saved histogram: {outpath}")
    return outpath

def standardize(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, float]:
    """Z-score με population std (ddof=0). Αν std==0 → όλα 0 και std=1.0."""
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x)); sigma = float(np.std(x, ddof=0))
    if sigma == 0.0:
        return (x - mu), mu, 1.0
    return (x - mu) / sigma, mu, sigma

# --------------------------
# Main
# --------------------------

def main():
    # --- CLI ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=str, help="CSV dataset path")
    ap.add_argument("--reports", default="reports/week6_data_quality_and_amount_scaling", type=str, help="Folder for JSON reports")
    ap.add_argument("--images", default="reports/figures/week6", type=str, help="Folder for plots")
    ap.add_argument("--out", default="data/data_interim/creditcard_amount_scaled.csv", type=str,
                    help="Προαιρετικό CSV με νέα στήλη Amount_scaled (για EDA)")
    ap.add_argument("--method", choices=["auto","zscore","logz"], default="auto",
                    help="auto: επιλέγει με βάση skewness (>1 → logz)")
    # Ποιότητα
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    ap.add_argument("--bins", default="auto", help="Histogram bins (π.χ. 'auto', 'fd', 100)")
    ap.add_argument("--images-format", default="png", choices=["png","svg"], help="Μορφή εικόνων")
    ap.add_argument("--require-cols", default="Amount", type=str,
                    help="Κόμμα-χωριστή λίστα από υποχρεωτικές στήλες (π.χ. 'Amount,Class')")
    ap.add_argument("--max-nulls", type=int, default=0, help="Ανώτατο επιτρεπτό σύνολο NaNs")
    ap.add_argument("--max-duplicates", type=int, default=0, help="Ανώτατες επιτρεπτές διπλότυπες γραμμές")
    ap.add_argument("--allow-negative-amounts", action="store_true",
                    help="Αν δοθεί, δεν αποτυγχάνει για αρνητικά Amount")
    ap.add_argument("--drop-duplicates", action="store_true",
                    help="Drop exact duplicate rows πριν τα quality checks")
    ap.add_argument("--dry-run", action="store_true",
                    help="Υπολογισμοί χωρίς write σε CSV/εικόνες/JSON")
    args = ap.parse_args()

    # --- Logging ---
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    global logger
    logger = logging.getLogger(__name__)
    logger.info("Ξεκίνησε: Week6 Data Quality & Amount Scaling")
    logger.debug("Args: %s", vars(args))

    # --- Paths ---
    data_path   = Path(args.data)
    reports_dir = Path(args.reports)
    images_dir  = Path(args.images)
    out_csv     = Path(args.out) if args.out else None
    images_ext  = f".{args.images_format}"

    if not args.dry_run:
        ensure_dir(reports_dir); ensure_dir(images_dir)
        if out_csv: ensure_dir(out_csv.parent)

    # --- Load ---
    logger.info(f"Φόρτωση: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Αποτυχία φόρτωσης CSV: {e}")
        raise

    # Προαιρετικά: ρίξε exact duplicates πριν τους ελέγχους
    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(ignore_index=True)
        dropped = before - len(df)
        logger.info(f"Έριξα {dropped} exact duplicate rows πριν τα quality checks.")

    # --- Required columns gate ---
    required_cols = [c.strip() for c in args.require_cols.split(",") if c.strip()]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Λείπουν υποχρεωτικές στήλες: {missing}")
        sys.exit(2)

    # --- Basic info ---
    shape = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    target_col = "Class" if "Class" in df.columns else None

    # --- Quality checks ---
    nulls_per_col = df.isna().sum().to_dict()
    total_nulls = int(np.sum(list(nulls_per_col.values())))
    logger.info(f"Σύνολο κενών: {total_nulls}")

    dup_count = int(df.duplicated().sum())
    logger.info(f"Διπλότυπες γραμμές: {dup_count}")

    invalid = {}
    if "Amount" in df.columns:
        invalid["amount_negative"] = int((df["Amount"] < 0).sum())
    if "Time" in df.columns:
        invalid["time_negative"] = int((df["Time"] < 0).sum())

    # --- Quality gates ---
    if total_nulls > args.max_nulls:
        logger.error(f"Nulls ({total_nulls}) > allowed ({args.max_nulls})."); sys.exit(2)
    if dup_count > args.max_duplicates:
        logger.error(f"Duplicates ({dup_count}) > allowed ({args.max_duplicates})."); sys.exit(2)
    if (not args.allow_negative_amounts) and invalid.get("amount_negative", 0) > 0:
        logger.error(f"Βρέθηκαν αρνητικά Amount ({invalid['amount_negative']}) και δεν επιτρέπονται."); sys.exit(2)

    # --- Amount profiling & scaling ---
    amount_stats: dict = {}
    method_chosen: str | None = None
    scaler_params: dict = {}
    scaled_hist_path = None
    amount_nans_filled = 0

    if "Amount" in df.columns:
        amount_raw = pd.to_numeric(df["Amount"], errors="coerce")
        amount_nans_filled = int(amount_raw.isna().sum())
        if amount_nans_filled > 0:
            logger.warning(f"Amount είχε {amount_nans_filled} NaNs — για EDA τα αντικαθιστώ με median.")
            amount = amount_raw.fillna(amount_raw.median())
        else:
            amount = amount_raw

        amount_stats = describe_amount(amount)
        skew = amount_stats.get("skew", 0.0)

        original_hist = images_dir / f"hist_amount_original{images_ext}"
        if not args.dry_run:
            plot_hist(amount, "Amount (original)", original_hist, bins=args.bins)

        if args.method == "auto":
            method_chosen = "logz" if (skew is not None and float(skew) > 1.0) else "zscore"
        else:
            method_chosen = args.method

        if method_chosen == "zscore":
            arr, mu, sigma = standardize(amount.to_numpy(np.float64))
            df["Amount_scaled"] = arr
            scaler_params = {"method": "zscore", "mean": mu, "std": sigma}
            scaled_hist_path = images_dir / f"hist_amount_zscore{images_ext}"
            if not args.dry_run:
                plot_hist(df["Amount_scaled"], "Amount (z-score)", scaled_hist_path, bins=args.bins)
        else:  # logz
            log_amount = np.log1p(amount.to_numpy(np.float64))
            arr, mu, sigma = standardize(log_amount)
            df["Amount_scaled"] = arr
            scaler_params = {"method": "logz", "pre": "log1p",
                             "mean_after_log": mu, "std_after_log": sigma}
            scaled_hist_path = images_dir / f"hist_amount_logz{images_ext}"
            if not args.dry_run:
                plot_hist(df["Amount_scaled"], "Amount (log1p → z-score)", scaled_hist_path, bins=args.bins)

        logger.info(f"Μέθοδος κλιμάκωσης που επιλέχθηκε: {method_chosen}")

    # --- Save preview CSV (EDA μόνο) ---
    if (not args.dry_run) and (out_csv is not None) and ("Amount_scaled" in df.columns):
        keep_cols = [c for c in ["Time","Amount","Amount_scaled","Class"] if c in df.columns]
        df_out = df[keep_cols].copy()
        df_out.to_csv(out_csv, index=False)
        logger.info(f"Αποθήκευση προεπισκόπησης κλιμάκωσης: {out_csv}")

    # --- Metadata / report ---
    import hashlib, time
    def file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""):
                h.update(chunk)
        return h.hexdigest()

    meta = {
        "source_path": str(data_path),
        "file_size_bytes": data_path.stat().st_size if data_path.exists() else None,
        "sha256": file_sha256(data_path) if data_path.exists() else None,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "log_level": args.log_level,
        "bins_used": args.bins,
        "images_format": args.images_format,
        "amount_nans_filled": amount_nans_filled
    }

    report = {
        "meta": meta,
        "shape": shape,
        "target": target_col,
        "nulls_total": total_nulls,
        "nulls_per_col": nulls_per_col,
        "duplicates": dup_count,
        "invalid_checks": invalid,
        "amount_profile": amount_stats,
        "scaling": {
            "chosen": method_chosen,
            "params": scaler_params,
            "images": {
                "original_hist": str(original_hist.as_posix()) if "Amount" in df.columns else None,
                "scaled_hist": str(scaled_hist_path.as_posix()) if scaled_hist_path else None
            }
        }
    }

    if not args.dry_run:
        ensure_dir(reports_dir)
        json_path = reports_dir / "06_data_quality_and_amount_scaling.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Αποθήκευση αναφοράς: {json_path}")

    logger.info("Τέλος: όλα καλά ✅  (Scaling για training μόνο μέσα σε scikit-learn Pipeline στο X_train)")

if __name__ == "__main__":
    main()
