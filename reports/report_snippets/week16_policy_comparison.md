# Week 16 — Threshold Policy Comparison (Validation-selected, Test locked)

This section compares **four threshold selection policies** for the XGBoost champion model.  
**Rule:** the threshold is selected on the **validation** set and then evaluated once on the **locked test** set.

---

## Test-set confusion matrices (by policy)

### 1) Cost-based (fp_cost=1, fn_cost=20) — **thr ≈ 0.0970**
- **TN=56633, FP=18, FN=18, TP=77**
- Precision = 77 / (77+18) = **0.8105**
- Recall = 77 / (77+18) = **0.8105**
- F1 = **0.8105** (equal to precision/recall due to FP=FN)
- Business view: balanced catch rate with moderate analyst load.

### 2) Precision constraint (precision ≥ 0.80) — **thr ≈ 0.1279**
- **TN=56635, FP=16, FN=18, TP=77**
- Precision = 77 / (77+16) = **0.8280**
- Recall = 77 / (77+18) = **0.8105**
- F1 ≈ **0.8191**
- Business view: keeps fraud catch the same as cost-based, while reducing false alarms.

### 3) Max F1 — **thr ≈ 0.9946**
- **TN=56650, FP=1, FN=30, TP=65**
- Precision = 65 / (65+1) = **0.9848**
- Recall = 65 / (65+30) = **0.6842**
- F1 ≈ **0.8076**
- Business view: extremely low false positives, but misses more frauds (higher FN).

### 4) Max F2 (max_fbeta, β=2) — **thr ≈ 0.9946**
- In this run, **Max F2 selected the same threshold as Max F1**, producing the same test confusion matrix:
  - **TN=56650, FP=1, FN=30, TP=65**
- Business view: same trade-off as above (very strict threshold).

---

## Summary table (Test set)

| Policy | Selected thr (VAL) | TP | FP | FN | TN | Precision | Recall | Comment |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cost-based (1,20) | ~0.0970 | 77 | 18 | 18 | 56633 | 0.8105 | 0.8105 | Good balance |
| Precision ≥ 0.80 | ~0.1279 | 77 | 16 | 18 | 56635 | 0.8280 | 0.8105 | **Best operational** (lower FP, same TP) |
| Max F1 | ~0.9946 | 65 | 1 | 30 | 56650 | 0.9848 | 0.6842 | Too many missed frauds |
| Max F2 (β=2) | ~0.9946 | 65 | 1 | 30 | 56650 | 0.9848 | 0.6842 | Same as Max F1 here |

---

## Final operating threshold (recommended)

**Chosen policy:** *Precision constraint (precision ≥ 0.80)*  
**Operating threshold:** **thr ≈ 0.1279** (validation-selected)  
**Why:** It preserves fraud catch performance (**TP=77**) while reducing false positives (**FP=16**) compared to cost-based (FP=18), lowering analyst workload and customer friction without sacrificing recall.

---

## Notes for the report

- Include the PR curve (validation) with the selected threshold point and the confusion matrix (test) for the **final policy**.
- Mention that all thresholds were selected on validation (no tuning on test), following a leakage-safe evaluation discipline.
