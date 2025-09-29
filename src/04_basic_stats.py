from pathlib import Path
from datetime import datetime
import pandas as pd

# ---- ΡΥΘΜΙΣΕΙΣ ----
# Μονοπάτι προς το αρχικό CSV (Kaggle creditcard.csv)
DATA_PATH = Path("data/data_raw/creditcard.csv")

# Πιθανά ονόματα για τη στήλη-στόχο (fraud label).
# Κάποια datasets τη γράφουν "Class", άλλα "class" ή "is_fraud".
TARGET_CANDIDATES = ["Class", "class", "is_fraud"]

# Φάκελος εξόδων όπου θα αποθηκευτεί το CSV με τα βασικά στατιστικά
OUT_DIR = Path("reports/week3_metrics")  # όπου θα σωθούν τα νούμερα

def find_target_column(df) -> str:
    """
    Επιστρέφει το όνομα της στήλης-στόχου από το DataFrame (π.χ. 'Class').
    Αν δεν βρεθεί κανένα από τα TARGET_CANDIDATES, σηκώνει KeyError.
    """
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # Αν φτάσουμε εδώ, δεν υπάρχει γνωστό όνομα στόχου στο dataset
    raise KeyError(
        f"Δεν βρέθηκε στήλη-στόχος. Ψάξαμε για: {TARGET_CANDIDATES}"
    )

def main():
    # 1) Έλεγχος ότι υπάρχει το dataset στο αναμενόμενο path
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Λείπει το {DATA_PATH}. Βάλε το dataset στον σωστό φάκελο."
        )

    # 2) Φόρτωση δεδομένων:
    #    Χρησιμοποιούμε 'usecols' με callable ώστε να διαβάσουμε ΜΟΝΟ στήλες
    #    που υπάρχουν και είναι στα TARGET_CANDIDATES (μειώνει μνήμη/χρόνο).
    #    Σημ.: το 'col is None' δεν χρειάζεται πρακτικά, αλλά δεν βλάπτει.
    df = pd.read_csv(
        DATA_PATH,
        usecols=lambda col: col in TARGET_CANDIDATES or col is None
    )

    # 3) Εντοπισμός της πραγματικής στήλης-στόχου στο DataFrame
    target = find_target_column(df)

    # 4) Μετατροπή της στήλης-στόχου σε ακέραιο τύπο (0/1).
    #    Αν έχει NaN ή μη-ακέραιες τιμές, εδώ θα σηκωθεί σφάλμα — που είναι καλό,
    #    γιατί προτιμούμε να αποτύχει νωρίς παρά να δώσουμε λάθος στατιστικά.
    y = df[target].astype("int32")

    # 5) Υπολογισμός βασικών μεγεθών
    total = len(y)                         # σύνολο γραμμών
    positives = int((y == 1).sum())        # πλήθος fraud συναλλαγών
    negatives = total - positives          # πλήθος non-fraud συναλλαγών
    rate_pct = positives / total * 100 if total > 0 else 0.0  # ποσοστό %

    # 6) Εκτύπωση φιλικής σύνοψης στην κονσόλα
    print(f"Target column : {target}")
    print(f"Total rows    : {total}")
    print(f"Fraud cases   : {positives}")
    print(f"Non-fraud     : {negatives}")
    print(f"Fraud rate    : {rate_pct:.4f}%")

    # 7) Αποθήκευση των αποτελεσμάτων σε CSV, ώστε να "κρατάμε τα νούμερα"
    OUT_DIR.mkdir(parents=True, exist_ok=True)  # δημιουργία φακέλου αν δεν υπάρχει

    # Timestamp της εκτέλεσης (για ιχνηλασιμότητα)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Γραμμή-ρεκόρ με όλα τα χρήσιμα πεδία
    row = {
        "timestamp":      ts,
        "data_path":      str(DATA_PATH),
        "target":         target,
        "total_rows":     total,
        "fraud_cases":    positives,
        "nonfraud_cases": negatives,
        "fraud_rate_pct": round(rate_pct, 6),  # 6 δεκαδικά για σταθερότητα
    }

    # Δημιουργούμε DataFrame μιας γραμμής και το γράφουμε σε CSV
    # (Με το default 'mode="w"' θα κάνει overwrite σε κάθε εκτέλεση.)
    pd.DataFrame([row]).to_csv(OUT_DIR / "basic_stats.csv", index=False)

if __name__ == "__main__":
    # Entry point: εκτέλεσε τη main όταν τρέχεις το αρχείο ως script
    main()
