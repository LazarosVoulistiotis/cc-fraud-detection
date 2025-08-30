---

# Ημέρα 1 — Jupyter & NumPy Fundamentals (2–3 ώρες)

## Στόχοι
- Να δουλεύω άνετα σε Jupyter.
- Να περάσω τα βασικά των NumPy arrays: δημιουργία, indexing, broadcasting, axis ops, reshapes.
- Να κρατάω καθαρές, χρήσιμες σημειώσεις.

---

## Checklist (γρήγορος έλεγχος)
- [ ] Άνοιξα venv και Jupyter
- [ ] Έφτιαξα notebook `notebooks/week2/day1_numpy.ipynb`
- [ ] Έτρεξα όλα τα cells για NumPy (δημιουργία, indexing, broadcasting, axis ops, reshape)
- [ ] Έλυνα mini-ασκήσεις (euclidean, z-score, dot, cov)
- [ ] Έγραψα “Gotchas” + σύντομα συμπεράσματα
- [ ] Έκανα daily git commit & push

---

## Βήματα

### 1) Jupyter basics (15’)
- **Άνοιγμα venv** (Windows):
  ```bash
  .\.venv\Scripts\activate.bat 
  *(σε CMD)*

## Shortcuts (τα πιο χρήσιμα):
- Εναλλαγή τύπου κελιού: M → Markdown, Y → Code
- Νέο κελί: Esc + A (πάνω), Esc + B (κάτω)
- Εκτέλεση: Shift + Enter
- Διαγραφή κελιού: DD
- Μετονομασία κελιού: Enter και γράψε

## NumPy τα απολύτως βασικά

1. Εισαγωγή & seed
- import numpy as np
Φορτώνει τη βιβλιοθήκη NumPy και της δίνει σύντομο ψευδώνυμο np (καθιερωμένο convention). Έτσι γράφεις np.array, np.mean, np.random κ.λπ.

- np.random.seed(42)
Ορίζει seed (σπόρο) για τον παλιό global γεννήτορα τυχαίων αριθμών του NumPy. Με απλά λόγια: από αυτή τη στιγμή και μετά, κλήσεις όπως np.random.rand(), np.random.randint() κ.ά. θα παράγουν την ίδια ακριβώς ακολουθία “τυχαίων” αριθμών κάθε φορά που τρέχεις το πρόγραμμα. Αυτό λέγεται αναπαραγωγιμότητα (reproducibility).

Γιατί να το χρησιμοποιήσεις:
- Για να έχεις σταθερά αποτελέσματα σε notebooks / πειράματα.
- Για να μπορεί κάποιος άλλος να αναπαράγει τα ίδια νούμερα με εσένα.

Συχνές απορίες & παγίδες:
- Το 42 είναι απλώς ένας αριθμός-παράδειγμα (διάσημο meme). Μπορεί να είναι οποιοσδήποτε ακέραιος στο εύρος 0 έως 2³²−1.
- Καλό είναι να κάνεις seed μία φορά στην αρχή. Αν το βάζεις μέσα σε loop, θα “ξαναρχίζει” η ίδια ακολουθία κάθε επανάληψη.
- Επηρεάζει μόνο το NumPy (np.random.*). Δεν επηρεάζει το random της Python ή τυχόν RNG άλλων βιβλιοθηκών (π.χ. PyTorch, TensorFlow) — αυτά θέλουν δικό τους seed.

2. Δημιουργία arrays
- np.array([...]) → χειροκίνητη λίστα σε NumPy array.
- np.arange(start, stop, step) → “range” με βήμα.
- np.linspace(start, stop, num) → ισαπέχουσες τιμές.
- np.zeros(shape) → μηδενικά.
- np.ones(shape) → άσσοι.
- np.random.rand(shape) → τυχαίοι δεκαδικοί [0,1).
- np.random.randint(low, high, size) → τυχαίοι ακέραιοι [low, high).

3. Ιδιότητες array
- dtype → τύπος δεδομένων (π.χ. float64).
- shape → διαστάσεις πίνακα (π.χ. 3x4).
- ndim → πόσες διαστάσεις έχει (εδώ 2D).
- size → πόσα στοιχεία έχει συνολικά.
- itemsize → πόσα bytes πιάνει κάθε στοιχείο στη μνήμη.

4. Indexing & Slicing
- arr[start:end:step] → slicing σε 1D.
- M[row, col] → indexing σε 2D.
- : → “πάρε όλα”.
- -1 → τελευταία θέση.
- Μπορείς να κόψεις υποπίνακες με slicing όπως M[0:2, 1:3]

5. Boolean masking & filtering
- Το mask είναι ένας πίνακας από True/False.
- Όταν το χρησιμοποιείς σαν δείκτη (vals[mask]), παίρνεις μόνο τα στοιχεία που αντιστοιχούν σε True.
- Αυτό λέγεται boolean indexing (ή filtering).
👉 Πολύ σημαντικό: το filtered εδώ είναι καινούργιο array (copy), όχι view. Αν αλλάξεις το filtered, δεν επηρεάζει το vals.

6. Broadcasting (πρόσθεση vector σε matrix, scaling)
- Broadcasting = όταν οι διαστάσεις 2 πινάκων δεν ταιριάζουν ακριβώς, το NumPy προσπαθεί να τις «τεντώσει» για να γίνουν συμβατές.
- A + v → broadcasting του v κατά μήκος των γραμμών.
- 2 * A → scalar πολλαπλασιασμός (το 2 γίνεται σαν να ήταν array ίδιου σχήματος με το A).

7. Συναρτήσεις κατά axis
- axis=0 → πράξη ανά στήλη.
- axis=1 → πράξη ανά γραμμή.
- Χωρίς axis, οι συναρτήσεις δουλεύουν σε όλο το array.

8. Reshape / Ravel / Flatten / 
- reshape → αλλάζει σχήμα (δεν αλλάζει δεδομένα).
- ravel() → flatten view (όπου γίνεται).
- flatten() → flatten copy (πάντα).
- concatenate → γενική συνένωση arrays.
- vstack = κάθετη στοίβαξη, hstack = οριζόντια στοίβαξη.

9. Vectorization vs Loops (%%timeit)
Η vectorization (πράξεις απευθείας σε NumPy arrays) είναι θεμέλιο στη Python για αριθμητικούς υπολογισμούς. Αποφεύγεις τα Python loops και εκμεταλλεύεσαι βελτιστοποιημένο C backend.

---

## 3.1 Ευκλείδεια απόσταση (χωρίς loop)
- np.linalg.norm(u-v) = Ευκλείδεια απόσταση μεταξύ u και v.
- Πολύ χρήσιμο σε ML (π.χ. KNN (K-Nearest Neighbors), clustering).
- Μπορείς και με άλλα norms: np.linalg.norm(u-v, ord=1) → Manhattan distance (calculate the distance between two points in a grid-like space, ord=np.inf → Chebyshev κ.ά.

## 3.2 Standardization (z‑score) ανά στήλη σε 2D array
- Έκανες Z-score standardization σε κάθε στήλη. ( A data transformation technique where each data point is converted into a z-score, which indicates how many standard deviations it is from the mean of the dataset. This process results in a standardized dataset with a mean of 0 and a standard deviation of 1. The main purpose of z-score standardization is to put data from different scales or units onto a common scale, making it easier to compare, analyze, and process. )
- Τώρα όλες οι στήλες έχουν ίδιο scale → απαραίτητο βήμα πριν από πολλά ML αλγορίθμους (π.χ. PCA, SVM, gradient descent).

## 3.3 Dot product, elementwise product, covariance matrix
- np.dot → scalar (ή matrix multiplication σε 2D).
- * → elementwise.
- np.cov → covariance matrix (features × features), χρήσιμο σε στατιστική, PCA (Principal component analysis), ML

# ✅ Mini-Ασκήσεις NumPy → ML Pipelines Checklist

| Mini-Άσκηση | Τεχνική που έμαθα | ML Αντίκρισμα | Σημειώσεις / Δικά μου Παραδείγματα |
|-------------|-------------------|---------------|------------------------------------|
| `np.random.seed` | Αναπαραγωγιμότητα | Σταθερά splits, ίδια init weights | [ ] |
| `np.arange`, `np.linspace`, `np.zeros`, `np.ones`, `np.random` | Δημιουργία arrays | Synthetic data, αρχικοποίηση βαρών | [ ] |
| Indexing / Slicing | Επιλογή υποσυνόλων | Train/test split, mini-batches, επιλογή features | [ ] |
| Boolean Masking | Φιλτράρισμα με συνθήκες | Επιλογή samples πάνω από threshold | [ ] |
| Broadcasting (`A+v`, `2*A`) | Αυτόματη ευθυγράμμιση | Bias add, feature scaling | [ ] |
| Στατιστικά (`mean`, `std`, `sum` με axis) | Aggregation ανά στήλη/γραμμή | Normalization, losses, batch statistics | [ ] |
| Reshape / Ravel / Flatten | Αλλαγή σχήματος δεδομένων | Εικόνες→vectors, flatten CNN layers | [ ] |
| Concatenate / Stack | Συνένωση arrays | Merge datasets, data augmentation | [ ] |
| Loop vs Vectorization | Απόδοση | Γρήγορο preprocessing μεγάλων datasets | [ ] |
| Euclidean Distance (`np.linalg.norm`) | Απόσταση διανυσμάτων | KNN, K-Means, similarity | [ ] |
| Standardization (z-score) | (x-μ)/σ | Scaling πριν training (SVM, NN, Logistic) | [ ] |
| Dot Product vs Elementwise | Διαφορά dot vs Hadamard | Cosine similarity, attention mechanisms | [ ] |
| Covariance Matrix (`np.cov`) | Συσχέτιση features | PCA, correlation analysis | [ ] |

## Συνήθη Gotchas
- Views vs Copies: Πολλά slicing/ravel() δίνουν view (οι αλλαγές αντικατοπτρίζονται στο αρχικό). Αν θέλεις ανεξάρτητο array, κάνε copy() ή flatten().
- Dtype upcasting: Πράξεις μεταξύ int/float μπορεί να αλλάξουν τύπο. Έλεγξε/όρισε dtype αν χρειάζεται.
- Broadcasting rules: Τα σχήματα “ευθυγραμμίζονται” από τα δεξιά. Αν μία διάσταση είναι 1 ή ίδια, γίνεται broadcast. Αλλιώς σφάλμα.
- Axis παρεξήγηση: axis=0 είναι κάθετες πράξεις ανά στήλη, axis=1 οριζόντιες ανά γραμμή.
- np.cov default: Από προεπιλογή θεωρεί κάθε γραμμή ως variable (rowvar=True). Για κλασικό (samples, features) χρειάζεται rowvar=False.

---

# Ημέρα 2 — Pandas Essentials (Series, DataFrame, I/O, Cleaning) (2–3 ώρες)

Goal της ημέρας: Να μάθεις τα βασικά του Pandas για φόρτωση/επιθεώρηση δεδομένων, επιλογές, φιλτράρισμα, δημιουργία νέων στηλών, missing values, groupby/agg, ταξινομήσεις και joins.

1) import pandas as pd
   import numpy as np

- Η pandas είναι το βασικό εργαλείο για πίνακες δεδομένων (DataFrames), ανάλογο με Excel αλλά πολύ πιο ισχυρό.
- NumPy = υπολογισμοί με πίνακες/διανύσματα/μήτρες και είναι η βάση πάνω στην οποία χτίζεται η pandas (κάθε στήλη DataFrame στην ουσία είναι NumPy array).

2) Επιλογές / Φιλτράρισμα / Νέες Στήλες 

-ΕΠΙΛΟΓΕΣ
- df["col"] → μία στήλη (Series).
- df[["col1","col2"]] → πολλές στήλες (DataFrame).
- df.loc[rows, cols] → επιλογή με labels (ονόματα).
- df.iloc[rows, cols] → επιλογή με index θέσεις (αριθμητικά).

-ΦΙΛΤΡΑΡΙΣΜΑ
- Χρησιμοποιούμε boolean indexing: df[ df["col"] > 100 ].
- Για πολλές συνθήκες: & → AND, | → OR, ~ → NOT (πάντα με παρενθέσεις).
- Ειδικές μέθοδοι: .isin([...]), .between(a,b), .str.contains("text") για πιο σύνθετα φιλτραρίσματα.

-MISSING VALUES
- df.isna().sum() → μετράει τα κενά ανά στήλη.
- .fillna(value) → αντικαθιστά NaN με μια τιμή (π.χ. "Unknown" ή 0).
- .dropna(subset=...) → πετάει γραμμές που έχουν NaN σε συγκεκριμένες στήλες.`

- Συνήθη Gotchas
- KeyError: αν η στήλη δεν υπάρχει (π.χ. λάθος όνομα).
- TypeError: αν amount δεν είναι αριθμητικό· μετέτρεψέ το:
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
- NaN στις ομαδοποιήσεις: συνήθως αγνοούνται στα aggregations, αλλά έλεγξε αν πρέπει να τα γεμίσεις/φιλτράρεις πριν.

4) Joins / Merge
- Με το pd.merge μπορείς να κάνεις joins μεταξύ πινάκων όπως στη SQL.
Σημειώσεις για joins
- inner: μόνο οι κοινοί merchants.
- left: κράτα όλα από df, πρόσθεσε merchants αν υπάρχουν.
- right: αντίστροφο.
- outer: όλα, γεμίζει NaN όπου δεν υπάρχει ταίριασμα.

Συγκεντρώνοντας τις σημειώσεις μου

## 🔹 Indexing & Selection
python
df["col"]          # μία στήλη (Series)
df[["c1","c2"]]    # πολλές στήλες (DataFrame)

df.loc[0:5, ["c1","c2"]]   # επιλογή με labels (inclusive)
df.iloc[0:5, 0:2]          # επιλογή με index θέσεις (exclusive στο τέλος)
👉 loc = με ονόματα, iloc = με αριθμητικές θέσεις.

🔹 GroupBy & Aggregations
df.groupby("merchant")["amount"].agg(["count","mean","sum"])
df.groupby("hour")["amount"].median().nlargest(5)
👉 Χρήσιμο για σύνοψη/στατιστικά ανά κατηγορία.

🔹 Joins (SQL-style)
- inner → κρατάει μόνο όσα ταιριάζουν και στα 2 DataFrames.
- left → κρατάει όλες τις γραμμές του αριστερού, NaN αν δεν υπάρχει match στο δεξί.
- right → ανάποδα από το left.
- outer → ένωση όλων, γεμίζει NaN όπου λείπουν τιμές.

🔹 Missing Values
df.isna().sum()                         # πόσα NaN ανά στήλη
df["col"] = df["col"].fillna("Unknown") # γέμισμα
df.dropna(subset=["merchant"], inplace=True)  # drop γραμμών
👉 Σκέψου: ποια NaN έχουν νόημα να συμπληρωθούν (impute) και ποια να διαγραφούν.

🔹 Συνήθη Σφάλματα
- KeyError: 'col' → η στήλη δεν υπάρχει (λάθος όνομα).
- SettingWithCopyWarning → απόπειρα αλλαγής σε αντίγραφο:
✔ Λύση → χρησιμοποίησε ανάθεση (df["col"] = ...) αντί για inplace=True σε στήλη.
- ValueError σε conversions → στήλη δεν είναι καθαρά αριθμητική/ημερομηνιακή.

🔹 ML Pipeline Connection
- Indexing / filtering → feature selection (π.χ. κράτα μόνο relevant features).
- GroupBy / aggregations → feature engineering (π.χ. μέσο ποσό ανά πελάτη).
- Joins → συνένωση διαφορετικών πηγών δεδομένων (transactions + merchants).
- Missing values → imputation βήμα στο preprocessing pipeline.
- Warnings & errors → σημαντικό να τα καταλάβεις γιατί μπορεί να “σπάσουν” pipelines σε παραγωγή.

# Ημέρα 3 — Plotting (Matplotlib + Seaborn) & Mini‑EDA

- Matplotlib (pyplot) = χαμηλού επιπέδου API για γραφήματα
- Seaborn = υψηλού επιπέδου API πάνω από Matplotlib με ωραίες default αισθητικές και «στατιστικά» γραφήματα (διανομές, box/violin, pairplot κ.λπ.).

“Συνταγές” κώδικα (γρήγορα snippets)
### Matplotlib savefig
plt.savefig("images/week2/plot.png", dpi=150, bbox_inches="tight")

### Seaborn histplot
sns.histplot(df["amount"], bins=50, kde=True)

### Boxplot
sns.boxplot(data=df, x="is_fraud", y="amount")

### Countplot
sns.countplot(data=df, x="is_fraud")

### Heatmap corr
sns.heatmap(df.corr(numeric_only=True), annot=False, cmap="viridis")

### 3–5 συμπεράσματα (παραδείγματα)
- Υπάρχει ανισορροπία κλάσεων: το is_fraud=1 είναι πολύ μικρό ποσοστό.
- Η κατανομή του amount είναι right‑skewed / long‑tail → χρήσιμο το log‑scale.
- Ο μέσος όρος ποσού διαφέρει ανά ώρα (πιθανά peaks σε συγκεκριμένες ώρες).
- Ασθενείς/μέτριες συσχετίσεις μεταξύ amount και άλλων numeric features (εξαρτάται από τα δεδομένα).

### Συνήθη Gotchas
- Figure overlap / κομμένες ετικέτες: Χρησιμοποίησε plt.tight_layout() πριν από savefig.
- Ανοιχτές φιγούρες: Μετά από savefig, κάνε plt.close() σε loops για να μην γεμίσει η μνήμη.
- Log scale: Απόφυγε log σε μη‑θετικές τιμές (0/αρνητικά) — καθάρισε/φίλτραρε πριν.
- numeric_only=True: Απαραίτητο σε df.corr όταν έχεις μη‑numeric στήλες.
- Ασυμφωνίες στον άξονα x: Για barplots/lineplots φρόντισε ταξινόμηση (sort_index, sort_values) για καλύτερη ανάγνωση.

### Self‑Check Quiz 

1. Πότε προτιμάς hist vs boxplot για μια μεταβλητή;
2. Τι κάνει plt.tight_layout() και γιατί είναι χρήσιμο πριν από savefig;
3. Πότε έχει νόημα plt.xscale("log");
4. Ποια είναι η βασική διαφορά plt.plot vs sns.lineplot;
5.Τι προσοχή θέλει όταν κάνεις heatmap συσχετίσεων σε DataFrame με strings/categoricals;

Γρήγορες απαντήσεις:

1. hist για μορφή κατανομής (skewness, modes), boxplot για outliers/median/IQR.
2. Ρυθμίζει spacing ώστε να μην κόβονται τίτλοι/labels → καθαρά exports.
3. Όταν η μεταβλητή έχει long‑tail/πολύ μεγάλες διακυμάνσεις.
4. Το plt.plot είναι low‑level Matplotlib, το sns.lineplot προσφέρει αισθητική & CI out‑of‑the‑box.
5. Χρησιμοποίησε numeric_only=True ή επίλεξε μόνο numeric στήλες.

---

# Ημέρα 4 — Κατέβασμα dataset (creditcard.csv) (30–60’)

## Στόχοι
- Να κατεβάσω και να αποθηκεύσω το ULB **Credit Card Fraud Detection** dataset στο `data/data_raw/`.
- Να ελέγξω ότι το CSV υπάρχει και είναι προσβάσιμο από τα scripts μου.

## Προϋποθέσεις
- **Kaggle account** και **API token** (`kaggle.json`).
- Ενεργό **virtual environment** (.venv) και Git repo έτοιμο.

## Δομή φακέλων (root project)
```
cc-fraud-detection/
├─ data/
│  ├─ data_raw/     # raw αρχεία όπως κατεβαίνουν
│  └─ data_work/    # προαιρετικός χώρος εργασίας/αντιγράφων
└─ ...
```

## Βήματα

### 1) Εγκατάσταση Kaggle CLI στο venv
```bat
pip install --upgrade pip
pip install kaggle
kaggle --version
```

### 2) Τοποθέτηση `kaggle.json` (Windows CMD)
```bat
mkdir "%USERPROFILE%\.kaggle"
copy "C:\Users\ΛΑΖΑΡΟΣ\Downloads\kaggle.json" "%USERPROFILE%\.kaggle\kaggle.json"
type "%USERPROFILE%\.kaggle\kaggle.json"   :: (προαιρετικό) πρέπει να έχει "username" και "key"
kaggle --version
```

> Αν δεις **401/Forbidden** κατά το κατέβασμα, άνοιξε από browser τη σελίδα του dataset και πάτησε “I Understand and Accept”.

### 3) Κατέβασμα στο `data/data_raw/`
Από **root** του project (`cc-fraud-detection>`):
```bat
mkdir data\data_raw 2>nul
kaggle datasets download -d mlg-ulb/creditcardfraud -p data\data_raw
```

### 4) Αποσυμπίεση
```bat
tar -xf data\data_raw\creditcardfraud.zip -C data\data_raw
```
Περιμένουμε να δούμε το αρχείο: `data/data_raw/creditcard.csv`

### 5) (Προαιρετικό) Αντίγραφο εργασίας
```bat
mkdir data\data_work 2>nul
copy /Y "data\data_raw\creditcard.csv" "data\data_work\creditcard.csv"
```

## Γρήγορος έλεγχος
```bat
dir data\data_raw\creditcard.csv
```
- Μέγεθος ~150MB
- Αναμενόμενο shape κατά τη φόρτωση: **(284807, 31)**

## Checklist
- [ ] Το Kaggle CLI λειτουργεί (`kaggle --version`).
- [ ] Έγινε download & unzip χωρίς σφάλματα.
- [ ] Υπάρχει το `data/data_raw/creditcard.csv`.
- [ ] (Προαιρετικό) Υπάρχει αντίγραφο στο `data/data_work/`.
- [ ] Daily **git commit & push**.

## Συχνά σφάλματα & λύσεις
- **401/Forbidden** → Συνδέσου στο Kaggle και αποδέξου τους όρους του dataset.
- **'kaggle' δεν αναγνωρίζεται** → Ενεργοποίησε `.venv` ή πρόσθεσε το script path στα περιβάλλοντα.
- **Zip δεν ανοίγει** → Ξανακατέβασε ή χρησιμοποίησε `powershell Expand-Archive`/7‑Zip.
- **Λάθος path** → Τρέξε εντολές από το **root** του project.


---

# Ημέρα 5 — Πρώτο script φόρτωσης (30–45’)

## Στόχοι
- Να επιβεβαιώσω ότι το dataset φορτώνει σωστά από το `data/data_raw/creditcard.csv`.
- Να εκτυπώσω **shape** και τις **10 πρώτες γραμμές**.
- Να κάνω **commit** το πρώτο μου loader script.

## Βήμα 1 — Δημιούργησε `src/01_load_data.py`
```python
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

> 💡 Τρέξε το από το **root** του project ώστε το σχετικό path να δουλέψει.

## Βήμα 2 — Εκτέλεση
```bat
python src/01_load_data.py
```
**Αναμενόμενο:** `Shape: (284807, 31)` και εκτύπωση των πρώτων 10 γραμμών.

## Βήμα 3 — Git Commit & Push
```bat
git add src/01_load_data.py
git commit -m "feat: add initial data loader (prints head & shape)"
git push
```

## Συχνά σφάλματα & λύσεις
- **FileNotFoundError** → Βεβαιώσου ότι τρέχεις από root, και ότι υπάρχει το `data/data_raw/creditcard.csv`.
- **ModuleNotFoundError: pandas** → Ενεργοποίησε `.venv` και `pip install pandas`.
- **UnicodeDecodeError/ParserError** → Δοκίμασε `pd.read_csv(DATA_PATH, encoding="utf-8", engine="python")` ή έλεγξε `sep`.
- **Μη αναμενόμενο shape** → Επιβεβαίωσε ότι φόρτωσες το σωστό αρχείο και όχι κάποιο άλλο CSV.
