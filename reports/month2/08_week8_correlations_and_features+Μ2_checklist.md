# Εβδομάδα 8 — Συσχετίσεις & Μηχανική Χαρακτηριστικών

## 1. Πίνακας συσχετίσεων (heatmap)
Υπολογίστηκε ο πίνακας συσχετίσεων για όλες τις αριθμητικές μεταβλητές και οπτικοποιήθηκε σε heatmap (βλ. `reports/figures/week8/corr_heatmap.png`). Όπως αναμενόταν, τα V1–V28 (PCA components) εμφανίζουν χαμηλές μεταξύ τους συσχετίσεις (σχεδόν γκρι φόντο). Παρατηρήθηκαν ελαφρώς πιο “θερμές” περιοχές ως προς την `Class` γύρω από V17, V14, V12, χωρίς όμως ισχυρές συστάδες μεταξύ features.

## 2. Συσχέτιση με την Class
Τα **top |corr|** με την κλάση (βλ. `reports/figures/week8/corr_with_target.png`) ήταν:
- **V17** ~ **−0.326**
- **V14** ~ **−0.303**
- **V12** ~ **−0.261**
- **V10** ~ **−0.217**
- **V16** ~ **−0.197**
- (έπονται: V3 −0.193, V7 −0.187, V11 +0.155, V4 +0.133, V18 −0.111 κ.λπ.)

Ακριβείς τιμές στο `reports/week8_corr_summary/week8_corr_with_target.csv` και στο JSON πεδίο `corr.top_corr_with_target_signed` (`reports/week8_corr_summary/week8_corr_summary.json`).  
*Σημ.*: Το `Amount_log1p` είναι νέο (engineered) feature και **δεν** περιλαμβάνεται στο παραπάνω ranking των original numeric columns.

## 3. Απλή μηχανική χαρακτηριστικών
- **Hour**: εξήχθη από το `Time` ως `(Time % 86400) // 3600`, και έγινε **κυκλική κωδικοποίηση**: `hour_sin`, `hour_cos`.
- **Amount_log1p**: λογαριθμικός μετασχηματισμός του Amount για σταθεροποίηση κατανομής.
Σημείωση: Το `Time` εκφράζει “δευτερόλεπτα από την 1η συναλλαγή”, άρα ο προσδιορισμός ώρας είναι proxy για 24ωρο μοτίβο και όχι πραγματική τοπική ώρα.

## 4. Fraud vs Hour
Το διάγραμμα `fraud_rate_by_hour.png` δείχνει το ποσοστό απάτης ανά ώρα ημέρας. Παρατηρήθηκε έντονο spike γύρω στις **02:00 (~1.7%)** και ένα δεύτερο peak κοντά στις **04:00 (~1.0%)**, ενώ μετά τις 06:00 το rate κινείται κοντά στο baseline (~0.173%). Οι τιμές και τα counts περιλαμβάνονται στο `reports/week8_corr_summary/week8_corr_summary.json` (`fraud_by_hour.rates`, `fraud_by_hour.counts`).

## 5. Σύνοψη Μήνα 2 (Exploratory Findings)
- **Κατανομές**: 
  - `Time`: εμφανίζει μοτίβο 24ώρου (proxy), χωρίς ακραίες αποκλίσεις.
  - `Amount`: έντονα skewed· ο μετασχηματισμός `log1p` βελτιώνει την κατανομή.
- **Class imbalance**: ~**0.17%** fraud → απαιτεί **PR-AUC**, **Recall-first** στόχευση, και τεχνικές **resampling** (π.χ. SMOTE).
- **Συσχετίσεις**: Οι PCA μεταβλητές είναι χαμηλά συσχετισμένες μεταξύ τους· ως προς `Class` ξεχωρίζουν κυρίως **V17, V14, V12, V10, V16**.
- **Νέα features**: `Hour`, `hour_sin`, `hour_cos`, `Amount_log1p`. Θα αξιολογηθούν σε baseline και επόμενα μοντέλα.

## 6. Προσχέδιο Μεθοδολογίας (Μήνας 3)
- **Models**: Logistic Regression (baseline + threshold tuning), Tree-based (Random Forest, XGBoost/LightGBM).
- **Imbalance**: Stratified split, PR-AUC ως βασική μετρική, **SMOTE** + εναλλακτικά undersampling. Threshold tuning για Recall ≥ 0.90 με αποδεκτό Precision.
- **Explainability & Ops**: Feature importances/SHAP, latency checks, reproducible pipelines (argparse, JSON/plots).

---

### Τι ακολουθεί (πλαίσιο Μήνα 3)
- Baseline **LogReg** με/χωρίς `Amount_log1p` και hour encodings.
- **PR-AUC** tracking, threshold tuning για **Recall ≥ 0.90**.
- **SMOTE/undersampling** συγκρίσεις.
- Προσθήκη **SHAP/feature importances** για business σχόλια.

### Milestone (End of Month 2)
Έχεις ολοκληρώσει αναλυτικό EDA και έχεις έτοιμο dataset για modeling. Τα βασικά βήματα προεπεξεργασίας (π.χ. scaling/cleaning) έχουν αποφασιστεί/υλοποιηθεί. Κατανοείς την έντονη ανισορροπία της κλάσης και έχεις τεκμηριώσει τα χαρακτηριστικά του dataset στο report. Επιπλέον, έχεις ολοκληρώσει ένα σχετικό online course/σεμινάριο πάνω σε data analysis ή ML (π.χ. “Data Analysis with Python” ή αντίστοιχο), ώστε να προστεθεί στα credentials σου.
