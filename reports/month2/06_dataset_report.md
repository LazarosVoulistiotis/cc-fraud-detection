---
title: "Dataset Description"
date: 2025-10-20
---

## Overview
Το dataset περιλαμβάνει συναλλαγές πιστωτικών καρτών Ευρωπαίων κατόχων και καλύπτει διάστημα δύο ημερών (Σεπτέμβριος 2013). Συνολικά περιέχει **283,726** συναλλαγές, εκ των οποίων **473** έχουν χαρακτηριστεί ως απάτη.
- **Ανισορροπία κλάσεων:** ~**0.1667%** θετική κλάση (fraud).
- **Χαρακτηριστικά:** V1–V28 (PCA), `Time`, `Amount`. Όλα αριθμητικά.

## Data Quality (Week 6)
- **Ελλείπουσες τιμές:** 0 συνολικά.
- **Διπλότυπα (μετά πολιτική):** 0 γραμμές.
- **Αρνητικά `Time`:** 0
- **Αρνητικά `Amount`:** 0

## Preprocessing Decision: Amount Scaling
- Επιλέχθηκε: **logz**.
- Skewness(Amount): **16.978803370060476**, Kurtosis: **844.4713194906487**.

## Plots
![hist_amount_original_nb.png](./hist_amount_original_nb.png)
![hist_amount_scaled_nb.png](./hist_amount_scaled_nb.png)