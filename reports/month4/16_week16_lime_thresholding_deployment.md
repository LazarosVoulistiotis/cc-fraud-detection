# Week 16 — LIME + Final Thresholding + Deployment Readiness (Month 4 Closure)

**Status:** ✅ Completed  
**Date:** 2026-02-23  
**Project:** Credit Card Fraud Detection (`cc-fraud-detection`)  

---

## 0) Executive summary

This week finalizes Month 4 by delivering:

- **LIME (Tabular) local explanations** for the same 3 case studies used in SHAP (TP/TN/Borderline).
- **Final threshold policy** selected on **validation** and evaluated once on the **locked test** set.
- **Final model selection narrative** and a deployment-ready conceptual architecture (API + future streaming).
- A clean “Week 16 deliverables + milestone closure” checklist for the report.

**Locked decision:**  
- **Champion model:** XGBoost  
- **Final threshold policy:** `precision_constraint_p80` (precision ≥ 0.80)  
- **Selected threshold (VAL):** `thr ≈ 0.1279`  
- **Test confusion matrix:** TP=77, FP=16, FN=18, TN=56635 (locked test set)

---

## 1) Goals of Week 16

1. **LIME (model-agnostic local explainability)** for on-demand analyst-style explanations.
2. **Final thresholding & metrics**: select threshold on validation using business-aligned criteria; report on locked test.
3. **Final model & deployment readiness narrative**: justify champion model and outline practical deployment design.
4. **Close Month 4 milestone**: “final model selected + interpretability results”.

---

## 2) Inputs / artifacts used

### 2.1 Model
- `models/xgb_week8.joblib` (champion artifact used for Week 15–16 runs)

### 2.2 Data splits (engineered schema)
- `data/data_interim/splits_week8/train.csv`
- `data/data_interim/splits_week8/val.csv`
- `data/data_interim/splits_week8/test.csv`

### 2.3 SHAP case indices (Week 15 reference)
- `reports/week15_shap/shap_cases.json`  
  - TP: idx=18427  
  - TN: idx=49260  
  - Borderline: idx=53293  

---

## 3) Part A — LIME (Tabular) local explainability

### 3.1 Implementation
**Script:** `src/16_lime_explainability.py`

**Command (Git Bash):**
```bash
python src/16_lime_explainability.py   --model-path models/xgb_week8.joblib   --data-train data/data_interim/splits_week8/train.csv   --data-test  data/data_interim/splits_week8/test.csv   --target-column Class   --shap-cases reports/month4/week15_shap/shap_cases.json   --figdir reports/figures/week16   --outdir reports/week16_lime   --threshold 0.0884   --num-features 10   --num-samples 5000   --seed 42
```

### 3.2 Outputs
**Figures:** `reports/figures/week16/`
- `lime_idx18427.png` (TP)
- `lime_idx49260.png` (TN)
- `lime_idx53293.png` (Borderline)

**Metadata:** `reports/week16_lime/`
- `lime_explanations.json`
- `lime_top_features.csv`
- `lime_config.json`

### 3.3 LIME vs SHAP (summary)
- Strong qualitative agreement on core drivers (notably V14/V12/V4/V10) for TP and borderline cases.
- Differences are expected near-threshold due to LIME’s local-linear surrogate and sampling sensitivity.

**Report snippet:** `reports/report_snippets/week16_lime.md`

---

## 4) Part B — Final thresholding & metrics (VAL-selected, TEST locked)

### 4.1 Script
**Script:** `src/16_thresholding_and_metrics.py`

### 4.2 Policies evaluated
We evaluated four validation-time policies and then reported performance on the locked test set:

- `cost_based` (fp_cost=1, fn_cost=20)
- `max_f1`
- `max_fbeta` (β=2 → Max F2)
- `precision_constraint` (precision ≥ 0.80)

### 4.3 Final policy (locked)
**Chosen policy:** `precision_constraint_p80`  
**Selected threshold (VAL):** `thr ≈ 0.1279`  
**Reason:** preserves fraud catch (TP=77) while lowering false positives (FP=16), reducing analyst workload/customer friction.

### 4.4 Outputs (per policy folders)
**Threshold outputs:** `reports/week16_thresholding/<policy>/`
- `final_metrics_test.json` / `final_metrics_test.csv`
- `threshold_selection.json`
- `val_threshold_sweep.csv`
- `config.json`

**Figures:** `reports/figures/week16/<policy>/`
- `pr_curve_val_with_threshold.png`
- `confusion_matrix_test.png`

**Report snippets:**
- `reports/report_snippets/week16_thresholding_metrics_snippet.md`
- `reports/report_snippets/week16_policy_comparison.md`

---

## 5) Part C — Final model selection & deployment readiness

### 5.1 Final model selection
The **XGBoost** model remains the champion:
- strong PR-AUC and MCC under severe class imbalance,
- supports audit-grade explainability via **TreeSHAP**,
- supports analyst-facing on-demand explainability via **LIME**.

### 5.2 Deployment readiness (conceptual)
- **Month 5 API**: Flask/FastAPI `POST /predict` (preprocessing + model scoring + threshold policy)
- **Future streaming**: Kafka → scoring microservice → alerting topic → analyst queue + monitoring

**Report snippet:** `reports/report_snippets/week16_final_model_and_deployment.md`

---

## 6) Week 16 deliverables & Month 4 milestone closure

**Deliverables checklist snippet:** `reports/report_snippets/week16_deliverables_and_milestone.md`

**Milestone achieved:**
> *Final model selected (XGBoost) + interpretability results delivered (SHAP + LIME) + operating threshold policy finalized (validation-selected precision constraint) and evaluated once on the locked test set.*

---

## 7) Next steps (Month 5 preview)

- Freeze final artifact as `models/xgb_final.joblib` (if retraining/freeze step is executed).
- Implement prediction API + reproducible preprocessing pipeline.
- Add monitoring hooks (logging, drift checks, threshold configuration).
- Optional streaming simulation (Kafka producer/consumer + alerting).

---
