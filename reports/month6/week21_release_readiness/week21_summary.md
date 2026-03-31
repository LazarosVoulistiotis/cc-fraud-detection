# Week 21 — Final Model Validation, API Stress Testing, and Release Readiness

## Overview
Week 21 functioned as the final validation and release-readiness pass of the project.
No new model training, threshold reselection, or serving-policy changes were introduced.
Instead, the work focused on verifying that the frozen fraud detection system is reproducible, technically validated, operationally interpretable, and ready for final presentation.

---

## 1. Freeze Check
The frozen serving setup was verified successfully.

### Frozen assets
- Model artifact: `models/xgb_final.joblib`
- Threshold config: `configs/threshold.json`
- Feature schema: `configs/feature_schema.json`
- Metadata config: `configs/model_metadata.json`

### Locked serving policy
- Policy name: `precision_constraint_p80`
- Policy version: `week16-locked`
- Selected on: validation
- Threshold: `0.1279`

### Frozen engineered features
- `Hour`
- `hour_sin`
- `hour_cos`
- `Amount_log1p`

This confirmed that Week 21 would evaluate the frozen system exactly as deployed, without altering the model or threshold policy.

---

## 2. Final Hold-Out Evaluation on Locked Test Set
The frozen XGBoost champion model was evaluated on the locked test set:

- Test set: `data/data_interim/splits_week8/test.csv`
- Threshold policy: `precision_constraint_p80`
- Threshold used: `0.1279`

### Final metrics
- Test size: **56,746**
- Fraud cases: **95**
- ROC-AUC: **0.96995**
- PR-AUC: **0.81713**
- Precision: **0.82796**
- Recall: **0.81053**
- F1-score: **0.81915**

### Confusion matrix
- TN = **56,635**
- FP = **16**
- FN = **18**
- TP = **77**

### Operational metrics
- Fraud alert rate: **0.001639**
- Alerts per 10k transactions: **16.39**
- Cost per transaction (FP=1, FN=20): **0.006626**

This exactly reproduced the expected locked-test result of the frozen serving setup.

---

## 3. Threshold Validation Without Leakage
A post-hoc sensitivity analysis was performed on the locked test predictions.
This analysis was reported strictly as comparison only and was **not** used to re-select a threshold.

### Thresholds compared
1. Final locked threshold: `0.1279`
2. Historical Week 12 reference threshold: `0.0884`
3. Post-hoc recall-first reference threshold (for sensitivity illustration only)

### Results summary

#### Final locked threshold `0.1279`
- Precision: **0.8280**
- Recall: **0.8105**
- F1: **0.8191**
- TP = **77**, FP = **16**, FN = **18**, TN = **56,635**
- Alerts/10k: **16.39**
- Cost/tx: **0.006626**

#### Historical threshold `0.0884`
- Precision: **0.7938**
- Recall: **0.8105**
- F1: **0.8021**
- TP = **77**, FP = **20**, FN = **18**, TN = **56,631**
- Alerts/10k: **17.09**
- Cost/tx: **0.006697**

#### Post-hoc recall-first reference
- Threshold: **0.000054**
- Precision: **0.0118**
- Recall: **0.9053**
- F1: **0.0233**
- TP = **86**, FP = **7,186**, FN = **9**, TN = **49,465**
- Alerts/10k: **1281.50**
- Cost/tx: **0.129807**

### Interpretation
The final locked threshold `0.1279` preserved fraud recall relative to the historical `0.0884` reference while reducing false positives, improving precision, and slightly lowering operational cost.
The recall-first post-hoc threshold achieved recall above 0.90, but only at the expense of an extreme increase in false positives and analyst workload.

---

## 4. Goal Assessment
Week 21 distinguished clearly between two different project goals.

### Goal A — Initial research goal
- Recall (Fraud) ≥ 0.90
- Precision ≥ 0.10

**Outcome:**  
The final serving threshold `0.1279` did **not** satisfy the original recall-first target, since recall remained at approximately **0.81**.
A post-hoc threshold could push recall above **0.90**, but precision collapsed to approximately **0.012**, making the operating point unrealistic.

### Goal B — Final serving goal
- frozen validation-selected serving threshold
- operationally acceptable precision
- fewer false positives
- lower analyst workload
- lower customer friction

**Outcome:**  
The frozen threshold `0.1279` satisfied the final business-aware serving goal better than the historical `0.0884` reference, because it preserved fraud recall while reducing false positives, improving precision, and slightly reducing cost.

### Final interpretation
This is not a weakness of the project.
It is a mature and explicitly justified business decision: the final system prioritizes a practical precision–recall balance for serving, rather than maximizing recall at any operational cost.

---

## 5. API Robustness and Edge-Case Testing
A final API-level robustness pass was conducted on the frozen FastAPI serving layer.

### Successful checks
- `GET /health` → success
- `GET /metadata` → success
- valid `POST /predict_by_id` → success
- invalid `row_id` → clean not-found error
- valid `POST /predict` → success
- very large `Amount` → accepted and scored successfully
- `Amount = 0` → accepted and scored successfully
- missing required field → deterministic validation rejection
- unexpected extra field → deterministic validation rejection
- negative `Amount` → deterministic validation rejection
- negative `Time` → deterministic validation rejection
- string instead of numeric → deterministic validation rejection

### Final hardening issue identified
Two non-finite numeric edge cases surfaced as `500 Internal Server Error` instead of a clean validation response:
- `NaN`
- `Infinity`

#### Technical interpretation
- In the `NaN` case, validation logic was triggered, but the custom validation error response failed during JSON serialization because the error payload contained `nan`.
- In the `Infinity` case, the preprocessing layer rejected the input explicitly with:
  `ValueError: Input contains NaN or infinite values.`

### Conclusion
The API is robust for valid requests and for most malformed inputs.
However, Week 21 identified one remaining serving-layer hardening gap for non-finite numeric payloads.

---

## 6. Peer / Friend User Testing
A peer-testing workflow was prepared in Week 21 to collect external usability feedback on the live demo.

### Prepared materials
- `reports/week21_user_testing/peer_feedback_log.csv`
- `reports/week21_user_testing/peer_testing_notes.md`

### Planned evaluation focus
- whether the tester understood what the demo does
- whether `fraud_probability` and `predicted_label` were clear
- whether any endpoint or output was confusing
- whether the system looked complete enough for portfolio or viva presentation

### Status
Peer feedback collection is pending and will be incorporated once received.

---

## 7. Revision Pass and Release Readiness
Week 21 also included a final revision pass focused on documentation and presentation readiness.

### Completed
- organized Week 21 evidence planning
- documented the `NaN` / `Infinity` hardening issue
- finalized wording for `/predict` and `/predict_by_id`
- drafted the Week 21 report subsection structure
- drafted the final Week 21 decision memo
- updated the Week 21 revision checklist

### Files prepared
- `reports/week21_release_readiness/revision_checklist.md`
- `reports/week21_release_readiness/evidence/evidence_manifest.md`
- `reports/week21_release_readiness/nan_inf_hardening_note.md`
- `reports/week21_release_readiness/endpoint_wording.md`
- `reports/week21_release_readiness/week21_report_structure.md`
- `reports/week21_release_readiness/week21_decision_memo.md`

---

## 8. Final Week 21 Conclusion
Week 21 confirmed that the frozen fraud detection system is ready to be presented as a deployment-oriented academic project.

The model artifact, serving schema, and threshold configuration were validated successfully.
The locked hold-out metrics were reproduced exactly.
The final threshold policy was justified against both research and operational goals.
The API was shown to be robust for valid requests and for most malformed inputs, while one remaining hardening issue (`NaN` / `Infinity`) was documented transparently.

Overall, Week 21 did not introduce a new model or a new threshold policy.
Instead, it provided final technical validation, evidence organization, documentation refinement, and presentation readiness for the deployed fraud detection system.
