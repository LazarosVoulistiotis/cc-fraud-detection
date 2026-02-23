# Week 16 — Deliverables & Month 4 Milestone Closure

This section lists the **Week 16 deliverables** and confirms completion status, with pointers to the exact artifacts generated in the repository.

---

## Week 16 Deliverables (Checklist)

| Deliverable | What it means | Status | Artifact(s) / Location |
|---|---|---:|---|
| **Final model** | Freeze the champion model for reporting and deployment. If additional tuning/retraining was performed, save as `xgb_final.joblib`. | ✅ / 🟡 | **Current champion:** `models/xgb_week8.joblib` (used in Week 15–16).  **Optional final retrain:** `models/xgb_final.joblib` (Month 5 / freeze step). |
| **Final threshold + metrics table** | Select threshold on **validation**, evaluate once on **locked test**; report Precision/Recall/F1/F2/MCC/PR‑AUC/ROC‑AUC + confusion matrix. | ✅ | **Policy locked:** `precision_constraint_p80` (precision ≥ 0.80), **thr ≈ 0.1279**.  Outputs per-policy under: `reports/week16_thresholding/<policy>/` including `final_metrics_test.json/.csv`, `threshold_selection.json`, `val_threshold_sweep.csv`. Figures under: `reports/figures/week16/<policy>/` (`pr_curve_val_with_threshold.png`, `confusion_matrix_test.png`). |
| **LIME plots + SHAP comparison** | Produce 3 LIME case explanations (TP/TN/Borderline) and compare against SHAP Week 15 waterfall cases. | ✅ | LIME: `reports/figures/week16/` (e.g., `lime_idx18427.png`, `lime_idx49260.png`, `lime_idx53293.png`) + `reports/week16_lime/` (`lime_explanations.json`, `lime_top_features.csv`, `lime_config.json`).  SHAP: `reports/figures/week15/` + `reports/week15_shap/shap_cases.json`.  Report snippet: `reports/report_snippets/week16_lime.md`. |
| **Explainability & industry relevance** | Explain why interpretability matters in fraud (auditability, analyst trust) and how SHAP+LIME address it. | ✅ | SHAP write-up: `reports/15_week15_shap_explainability.md` + snippet `reports/report_snippets/week15_shap.md`.  LIME+comparison snippet: `reports/report_snippets/week16_lime.md`.  Final model + deployment narrative: `reports/report_snippets/week16_final_model_and_deployment.md`. |
| **Month 4 milestone** | Close Month 4 with: “final model selected + interpretability results” (model choice + threshold policy + explainability evidence). | ✅ | Milestone statement included in: `reports/report_snippets/week16_final_model_and_deployment.md` and this checklist. |

---

## Final operating choice (locked for the report)

- **Champion model:** XGBoost (tree-based, high PR-AUC, strong MCC, explainable via SHAP & LIME)  
- **Threshold policy:** **precision constraint** (precision ≥ 0.80)  
- **Selected threshold (VAL):** **thr ≈ 0.1279**  
- **Locked test reporting:** confusion matrix + metrics table generated under `reports/week16_thresholding/precision_constraint_p80/`.

---

## Month 4 closure statement (copy-paste)

> **Month 4 Milestone achieved:** *Final model selected (XGBoost champion) + interpretability results delivered (SHAP global/local + LIME case studies) + operating threshold policy finalized (validation-selected precision constraint) and evaluated once on the locked test set.*

---
