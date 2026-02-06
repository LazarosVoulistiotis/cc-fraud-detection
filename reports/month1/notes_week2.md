# Week 2 — Python for Data Science (NumPy, Pandas, Plotting, Dataset Setup)

---

# Ημέρα 1 — Jupyter & NumPy Fundamentals (2–3 ώρες)

## Στόχοι
- Να δουλεύω άνετα σε Jupyter.
- Να περάσω τα βασικά των NumPy arrays: δημιουργία, indexing, broadcasting, axis ops, reshapes.
- Να κρατάω καθαρές, χρήσιμες σημειώσεις.

---

## Checklist
- [ ] Άνοιξα venv και Jupyter
- [ ] Έφτιαξα notebook `notebooks/week2/day1_numpy.ipynb`
- [ ] Έτρεξα όλα τα cells για NumPy (δημιουργία, indexing, broadcasting, axis ops, reshape)
- [ ] Έλυνα mini-ασκήσεις (euclidean, z-score, dot, cov)
- [ ] Έγραψα “Gotchas” + σύντομα συμπεράσματα
- [ ] Daily git commit & push

---

## NumPy βασικά σημεία
- **Seed:** `np.random.seed(42)` → reproducibility (ίδιοι τυχαίοι αριθμοί κάθε φορά).
- **Δημιουργία arrays:** `np.array`, `np.arange`, `np.linspace`, `np.zeros`, `np.ones`, `np.random`.
- **Ιδιότητες:** `dtype`, `shape`, `ndim`, `size`, `itemsize`.
- **Indexing/slicing:** `arr[0:5]`, `M[0:2, 1:3]`.
- **Boolean masking:** `vals[mask]`.
- **Broadcasting:** π.χ. `A + v`, `2*A`.
- **Axis ops:** `axis=0` (στήλες), `axis=1` (γραμμές).
- **Reshape/Flatten:** `.reshape()`, `.ravel()`, `.flatten()`.
- **Vectorization:** γρήγορες πράξεις χωρίς Python loops.

### Mini-ασκήσεις
- **Euclidean distance:** `np.linalg.norm(u-v)` → χρήσιμο σε KNN.
- **Standardization (z-score):** `(x-μ)/σ` → scaling πριν από ML.
- **Dot product / covariance matrix:** dot = similarity, cov = συσχέτιση features.

---

# Ημέρα 2 — Pandas Essentials (2–3 ώρες)

## Στόχοι
- Να μάθω βασικά Pandas: φόρτωση/επιθεώρηση δεδομένων, επιλογές, φιλτράρισμα, νέες στήλες, missing values, groupby/agg, ταξινομήσεις και joins.

---

## Pandas βασικά σημεία
```python
import pandas as pd
import numpy as np
df["Amount"]               # μία στήλη (Series)
df[["Time","Amount"]]      # πολλές στήλες
df.loc[0:5, ["Amount","Class"]]  # με labels
df.iloc[0:5, 0:2]          # με index θέσεις 
```
## Φιλτράρισμα
```df[df["Amount"] > 100]
df[(df["Amount"] > 100) & (df["Class"] == 1)]
```
## Missing values
```df.isna().sum()
df.fillna(value)
df.dropna(subset=["col"])
```
## GroupBy / Aggregations
df.groupby("Class")["Amount"].mean()
df.groupby("Class")["Amount"].median()

## Joins
```pd.merge(df1, df2, on="key", how="left")
```
## Συνήθη Gotchas
- KeyError: λάθος όνομα στήλης (χρησιμοποίησε print(df.columns)).
- numeric_only: σε df.corr() βάλε numeric_only=True.
- SettingWithCopyWarning: απόφυγε .loc σε slices χωρίς copy.

---

# Ημέρα 3 — Plotting (Matplotlib + Seaborn) & Mini-EDA (2–3 ώρες)

## Στόχοι
- Να παράγω καθαρά γραφήματα (διανομές, boxplots, bar/line).
- Να αποθηκεύω εικόνες σε φάκελο reports/figures/week2/.
- Να κάνω μια μικρή EDA στο creditcard.csv.

## Παραδείγματα
```import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set_theme()
```
## Histogram:
``` sns.histplot(df["Amount"], bins=50, kde=False)
```
## Boxplot:
``` sns.boxplot(data=df, x="Class", y="Amount")
```
## Countplot (class distribution):
``` sns.countplot(data=df, x="Class")
```
## Correlation heatmap:
``` sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
```
## Insights (Mini-EDA)
- Fraud rate: ~0.173% → πολύ έντονη ανισορροπία.
- Κατανομή Amount: right-skewed με λίγα πολύ μεγάλα amounts → λογαριθμική κλίμακα βοηθάει.
- Boxplot by Class: Fraud συναλλαγές (Class=1) τείνουν να έχουν ελαφρώς μεγαλύτερα amounts.
- Heatmap: Καμία ισχυρή συσχέτιση του Amount με άλλα features· ασθενής θετική με Class.

---

# Ημέρα 4 — Κατέβασμα dataset (30–60’)

## Στόχοι
- Να κατεβάσω το ULB Credit Card Fraud Detection dataset.
- Να το αποθηκεύσω στο data/data_raw/.

## Βήματα
- Kaggle CLI setup με kaggle.json.
- kaggle datasets download -d mlg-ulb/creditcardfraud -p data/data_raw
- tar -xf data/data_raw/creditcardfraud.zip -C data/data_raw
- Έλεγχος: creditcard.csv υπάρχει, μέγεθος ~150MB, shape (284807, 31).

---

# Ημέρα 5 — Πρώτο script φόρτωσης (30–45’)

## Στόχοι
- Να φορτώσω το dataset με script.
- Να δω shape + πρώτες 10 γραμμές.

## Script: src/01_load_data.py
```
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/data_raw/creditcard.csv")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Put the dataset in data/data_raw/")
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print(df.head(10).to_string())

if __name__ == "__main__":
    main()
```

---

# ✅ Συμπέρασμα Εβδομάδας 2
- Έμαθα τα βασικά NumPy (arrays, vectorization).
- Έμαθα Pandas essentials (DataFrames, indexing, groupby, missing values).
- Έκανα plotting & mini-EDA στο creditcard.csv.
- Κατέβασα το dataset και έφτιαξα πρώτο loader script.
- Έκανα καθημερινά commits → repo αρχίζει να είναι “business-ready”.
