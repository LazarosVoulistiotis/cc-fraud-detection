"""
01_load_data.py
---------------
Αρχικό script φόρτωσης δεδομένων για το project "Credit Card Fraud Detection".

Λειτουργίες:
- Ελέγχει αν υπάρχει το dataset (creditcard.csv) στον φάκελο data/data_raw/
- Διαβάζει το CSV σε pandas DataFrame
- Εμφανίζει το σχήμα του dataset (γραμμές, στήλες)
- Εμφανίζει δείγμα γραμμών (default = 10)

Σκοπός:
Βασικός έλεγχος για να βεβαιωθούμε ότι το dataset είναι διαθέσιμο και σωστά φορτωμένο.
"""

import pandas as pd
from pathlib import Path
import argparse
import logging

# ---- Logging config ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Default dataset path
DATA_PATH = Path("data/data_raw/creditcard.csv")


def load_data(path: Path, n_rows: int = 10) -> pd.DataFrame:
    """Φορτώνει το dataset και επιστρέφει DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put the dataset in data/data_raw/")

    df = pd.read_csv(path)
    logging.info(f"Dataset loaded: {path}")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Preview (first {n_rows} rows):\n{df.head(n_rows).to_string()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Load and preview credit card fraud dataset")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Path to dataset CSV (default: data/data_raw/creditcard.csv)")
    parser.add_argument("--rows", type=int, default=10,
                        help="Number of rows to preview (default: 10)")
    args = parser.parse_args()

    path = Path(args.data)
    load_data(path, args.rows)


if __name__ == "__main__":
    main()
