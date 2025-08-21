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