---
title: "Dataset Description"
date: 2025-10-20
---

## Overview
Το dataset περιλαμβάνει συναλλαγές πιστωτικών καρτών Ευρωπαίων κατόχων και καλύπτει διάστημα δύο ημερών (Σεπτέμβριος 2013).
Μετά την αφαίρεση exact duplicates περιέχει **283726 συναλλαγές**, εκ των οποίων **473** έχουν χαρακτηριστεί ως απάτη.
- **Ανισορροπία κλάσεων:** ~**0.1667%** θετική κλάση (fraud) — έντονα μη ισορροπημένο.
- **Χαρακτηριστικά:** `V1–V28` (PCA), `Time`, `Amount`. Όλα αριθμητικά.

## Data Quality (Week 6)
- **Missing values:** 0 συνολικά.
- **Duplicates μετά την πολιτική:** 0 γραμμές.
- **Αρνητικά `Time`:** 0
- **Αρνητικά `Amount`:** 0

## Preprocessing Decision — Amount Scaling
Η μεταβλητή `Amount` παρουσιάζει ισχυρή ασυμμετρία/βαριά ουρά.
- **Skewness:** **16.9788**
- **Kurtosis:** **844.4713**
- **Επιλογή μεθόδου:** **logz**

**Scaler parameters (για αναπαραγωγιμότητα):**
- **mean:** `3.15376049`
- **std:** `1.65707708`  
_Parameters refer to values **after** `log1p(Amount)`._

> *Practice note:* Στην εκπαίδευση μοντέλων, ο ίδιος μετασχηματισμός εφαρμόζεται **εντός scikit-learn Pipeline** και γίνεται **fit μόνο στο `X_train`** (αποφυγή data leakage).

## Plots
**Κατανομή πριν/μετά:**
![Amount (original)](./hist_amount_original_nb.png)
![Amount (log1p → z-score)](./hist_amount_scaled_nb.png)

## Παραδοτέα
- **JSON report:** ποιοτικοί έλεγχοι, περιγραφικά, μέθοδος κλιμάκωσης, metadata.
- **CSV προεπισκόπησης (EDA-only):** `Time`, `Amount`, `Amount_scaled`, `Class`.
- **Plots:** ιστογράμματα πριν/μετά.

