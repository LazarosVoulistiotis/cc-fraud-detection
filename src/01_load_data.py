"""
01_load_data.py
---------------
Αρχικό script φόρτωσης δεδομένων για το project "Credit Card Fraud Detection".

Λειτουργίες:
- Ελέγχει αν υπάρχει το dataset (creditcard.csv) στον φάκελο data/data_raw/
- Διαβάζει το CSV σε pandas DataFrame
- Εμφανίζει το σχήμα του dataset (γραμμές, στήλες)
- Εμφανίζει τις 10 πρώτες γραμμές σε μορφή πίνακα

Σκοπός:
Βασικός έλεγχος για να βεβαιωθούμε ότι το dataset είναι διαθέσιμο και σωστά φορτωμένο.
"""

import pandas as pd            # Βιβλιοθήκη για επεξεργασία και ανάλυση δεδομένων
from pathlib import Path        # Για ασφαλή διαχείριση paths ανεξαρτήτως λειτουργικού συστήματος

# Διαδρομή του dataset
DATA_PATH = Path("data/data_raw/creditcard.csv")


def main():
    """
    Κύρια συνάρτηση που εκτελεί το script φόρτωσης.
    - Ελέγχει αν το dataset υπάρχει
    - Φορτώνει το dataset σε pandas DataFrame
    - Εκτυπώνει shape και δείγμα δεδομένων
    """
    # Έλεγχος αν το dataset υπάρχει στο συγκεκριμένο path
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Put the dataset in data/data_raw/"
        )

    # Φορτώνουμε το dataset σε DataFrame
    df = pd.read_csv(DATA_PATH)

    # Εκτυπώνουμε μέγεθος (γραμμές, στήλες)
    print("Shape:", df.shape)

    # Εμφανίζουμε τις 10 πρώτες γραμμές του dataset
    print(df.head(10).to_string())


# Το παρακάτω μπλοκ διασφαλίζει ότι η main() εκτελείται
# μόνο όταν τρέχουμε το αρχείο απευθείας (όχι αν γίνει import)
if __name__ == "__main__":
    main()
