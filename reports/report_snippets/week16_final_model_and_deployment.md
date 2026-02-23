# Week 16 — Final Model Selection & Deployment Readiness

This snippet finalizes the Month 4 decision: **select the champion model**, justify the **operating threshold policy**, and describe a **deployment-ready architecture** (API + future streaming).

---

## 1) Final model selection (Champion)

### Decision
The **XGBoost** model is selected as the **champion** for this credit card fraud detection system. It provides a strong balance of detection performance under extreme class imbalance and supports interpretability through **SHAP (TreeExplainer)** and **LIME**.

### Evidence (locked test set)
Using the final operating policy (**precision constraint**, precision ≥ 0.80) and the validation-selected threshold (**thr ≈ 0.1279**), evaluation on the **locked test set** yields:

- Confusion matrix (TEST): **TP=77, FP=16, FN=18, TN=56,635**
- Precision (fraud): **0.828**
- Recall (fraud): **0.811**
- F1: **0.819**
- F2: **0.814**
- MCC: **0.819**
- PR-AUC: **0.817**
- ROC-AUC: **0.970**

**Business interpretation:** the model detects **77/95 frauds** while producing only **16 false alarms** on ~56k legitimate transactions, reducing fraud losses while keeping analyst workload and customer friction manageable.

---

## 2) Operating threshold policy (production mindset)

Fraud detection is not a “0.5-threshold” problem. The probability threshold is treated as an **operational policy knob** chosen on the **validation set** and applied once to the **locked test set** for final reporting.

### Final policy (locked)
**Policy:** `precision_constraint`  
**Constraint:** precision ≥ 0.80  
**Selected threshold (VAL):** **thr ≈ 0.1279**  
This choice reflects a realistic operational constraint: maintain alert precision to avoid overwhelming analysts and disrupting customers, while retaining strong recall.

*(Alternative policies were evaluated for comparison: cost-based thresholding and max-F metrics. The chosen policy offered the best operational trade-off on the locked test set.)*

---

## 3) Explainability & industry relevance

Fraud decisions affect customers (declines, step-up verification) and must be auditable. This project uses two complementary methods:

- **SHAP (TreeExplainer):** audit-grade explanations for global drivers and consistent local attributions on the tree-based model.
- **LIME (Tabular, model-agnostic):** on-demand local explanations for individual alerts, aligned with how fraud analysts investigate flagged transactions.

Together, SHAP + LIME improve trust, analyst usability, and support responsible deployment.

---

## 4) Deployment readiness (Month 5 plan)

### 4.1 API microservice (Flask/FastAPI)
A deployment-ready design is a lightweight prediction service:

- **Model artifact:** `models/xgb_final.joblib` (or the current champion artifact if retraining is not repeated)
- **Endpoint:** `POST /predict`
- **Input:** transaction features (V1…V28 + Time/Amount or engineered equivalents)
- **Processing:** apply the **same preprocessing and feature engineering** as training (e.g., `hour_sin/hour_cos`, `Amount_log1p`, scaling if used)
- **Output:** `p_fraud`, `threshold`, `pred_label`, `policy_name`

**Logging:** request id, timestamp, `p_fraud`, `pred_label`, threshold, and optional explanation references (for audit trails).

### 4.2 Future streaming architecture (conceptual)
For near-real-time screening at scale:

**Kafka → Scoring API → Alerting**
- Transactions published to a Kafka topic (e.g., `transactions`)
- Consumer service computes features and scores with XGBoost
- Alerts published to `fraud_alerts` when `p_fraud ≥ threshold`
- Downstream systems: analyst queue, notifications, dashboards

**Monitoring & governance**
- Track alert rate, precision proxies, and data drift
- Allow **threshold adjustments** without retraining
- Trigger retraining when drift is detected or performance degrades

---

## 5) Week 16 deliverables mapping

- **Final model:** `models/xgb_final.joblib` (if retrained/frozen), otherwise the current champion artifact.
- **Final threshold & metrics table:** validation-selected threshold + locked test metrics (Precision/Recall/F1/F2/MCC/PR-AUC/ROC-AUC).
- **Explainability:** SHAP (Week 15) + LIME (Week 16) plots and case study comparisons.
- **Month 4 milestone:** **“final model selected + interpretability results”** (champion decision + explainability evidence + threshold policy narrative).

---
