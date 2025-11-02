# Week 8 — Insights (Correlations & Feature Engineering)

**Dataset snapshot:** 284,807 γραμμές × 31 στήλες • **Fraud rate (overall):** 0.1727%

---

## 1) Heatmap συσχετίσεων (Pearson)
- Οι **V1–V28 (PCA)** είναι σχεδόν ασυσχέτιστες μεταξύ τους → **χαμηλός κίνδυνος πολυσυγγραμμικότητας**.
- Η `Class` εμφανίζει **ασθενείς/μέτριες** συσχετίσεις με ορισμένα components (κλασικό σε fraud, όπου οι σχέσεις είναι συχνά μη γραμμικές).
- 📎 **Artifact:** `reports/figures/week8/corr_heatmap.png`

---

## 2) Top |corr| με Class (signed για ερμηνεία κατεύθυνσης)
*(απόλυτη τιμή συσχέτισης: μεγαλύτερη ⇒ πιο “σχετικό” feature για αρχική ιεράρχηση)*

- **V17** ≈ **−0.326**
- **V14** ≈ **−0.303**
- **V12** ≈ **−0.261**
- **V10** ≈ **−0.217**
- **V16** ≈ **−0.197**
- **V3**  ≈ **−0.193**
- **V7**  ≈ **−0.187**
- **V11** ≈ **+0.155**
- **V4**  ≈ **+0.133**
- **V18** ≈ **−0.111**

💡 **Ερμηνεία:** Αρνητικό πρόσημο ⇒ μεγαλύτερες τιμές του feature τείνουν να συνδέονται γραμμικά με **μικρότερη** πιθανότητα `Class=1` (και αντίστροφα).  
⚠️ **Σημείωση:** Συσχέτιση ≠ αιτιότητα. Οι τιμές είναι μέτριες (0.1–0.33) λόγω έντονης ανισορροπίας — είναι **starting point**, όχι τελική κρίση.  
📎 **Artifact:** `reports/figures/week8/corr_with_target.png`  
📎 **CSV (signed corr προς target):** `reports/week8_corr_with_target.csv`

---

## 3) Χρονικό μοτίβο — Fraud rate ανά ώρα (0–23)
- **Spike ~02:00** ≈ **1.71%** (~**10×** πάνω από baseline 0.1727%).
- **Δεύτερο peak ~04:00** ≈ **~1.0%** (~**6×** πάνω από baseline).
- Μετά τις 06:00 κινείται κοντά στο baseline με μικρές διακυμάνσεις.  
📎 **Artifact:** `reports/figures/week8/fraud_rate_by_hour.png`  
📎 **JSON σύνοψη (rates & counts):** `reports/week8_corr_summary.json` → `fraud_by_hour.rates` & `fraud_by_hour.counts`

> ✅ **QA:** Χρησιμοποίησε τα **counts** ανά ώρα (στο JSON) για να επιβεβαιώσεις ότι τα spikes δεν προκύπτουν από ελάχιστα δείγματα.

---

## 4) Νέα features (για μοντέλα & κανόνες)
- `Hour` (0–23)  
- **Κυκλική κωδικοποίηση**: `hour_sin`, `hour_cos` (καλύτερη αναπαράσταση periodicity)  
- `Amount_log1p` (μείωση skew, σταθεροποιεί εκπαίδευση)  
📎 **Interim dataset:** `data/data_interim/creditcard_features_week8.csv`

---

## Business αξία & ενέργειες

**Αξία**
- Γρήγορη ιεράρχηση features για **baseline μοντέλα** και **αναφορά**.
- **Time-aware** insight: συγκεκριμένες νυχτερινές ώρες είναι δυσανάλογα ριψοκίνδυνες → υλικό για **adaptive thresholds** & **operational staffing**.
- Νέα features έτοιμα για pipeline/feature store.

**Προτεινόμενες ενέργειες (άμεσα)**
1. **Modeling με/χωρίς** (`Hour`, `hour_sin`, `hour_cos`, `Amount_log1p`) και σύγκριση **PR-AUC** / Recall@FPR=1%–5%.
2. **Mutual Information (MI)** και **μοντελο-βασικές σημασίες** (π.χ. SHAP) για να πιάσουμε **μη γραμμικές** σχέσεις.
3. **Time-aware thresholding**: Αυστηρότερα όρια/κανόνες 01:00–05:00 (δοκιμή σε validation).
4. **Monitoring**: Λογική για **drift** σε hourly rates (trigger alerts αν απόκλιση > x× baseline).

---

## Αναπαραγωγή

**Bash / Git Bash / WSL / macOS / Linux**
```bash
python src/08_correlations_and_features.py \
  --data data/data_raw/creditcard.csv \
  --target Class \
  --reports reports \
  --images reports/figures/week8 \
  --out data/data_interim/creditcard_features_week8.csv \
  --top-n 15 \
  --corr-method pearson \
  --corr-sample 0 \
  --mi-topn 25
```

---

## Caveats (επιφυλάξεις)

- Η **Pearson corr** αποτυπώνει **γραμμικές** σχέσεις. Συμπλήρωσέ τη με **Spearman/MI/SHAP.**
- **Class imbalance:** Μικρές corr δεν σημαίνουν “ασήμαντα features”. Το signal στο fraud είναι συχνά λεπτό.
- **Hourly spikes:** να τεκμηριώνονται μαζί με **counts** και interval uncertainty (π.χ. Wilson CI) αν τα παρουσιάσεις σε stakeholders.