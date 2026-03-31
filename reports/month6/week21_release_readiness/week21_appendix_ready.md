# Appendix — Week 21 Supporting Evidence

## Appendix A — Freeze and Hold-Out Evaluation Evidence

### A.1 Frozen serving configuration
Week 21 verified that the final evaluation used the frozen serving stack exactly as deployed. The frozen system consisted of:
- model artifact: `models/xgb_final.joblib`
- threshold configuration: `configs/threshold.json`
- feature schema: `configs/feature_schema.json`

The locked serving policy remained:
- policy name: `precision_constraint_p80`
- policy version: `week16-locked`
- threshold: `0.1279`

This confirms that Week 21 did not introduce a new model artifact, a new feature schema, or a new serving threshold.

### A.2 Locked hold-out evaluation
The frozen XGBoost champion model was evaluated on:
- `data/data_interim/splits_week8/test.csv`

The final hold-out evaluation reproduced the expected locked-test results exactly:
- ROC-AUC = 0.96995
- PR-AUC = 0.81713
- Precision = 0.82796
- Recall = 0.81053
- F1-score = 0.81915

Confusion matrix:
- TN = 56,635
- FP = 16
- FN = 18
- TP = 77

Operational metrics:
- fraud alert rate = 0.001639
- alerts per 10,000 transactions = 16.39
- cost per transaction (FP = 1, FN = 20) = 0.006626

These results confirm that the frozen model, frozen schema, and locked threshold configuration are internally aligned and reproducible.

---

## Appendix B — Threshold Sensitivity Evidence

### B.1 Scope of the analysis
Threshold analysis in Week 21 was reported strictly as post-hoc sensitivity analysis and not as a new threshold selection step. The locked test set was not used to re-define serving policy.

### B.2 Thresholds compared
The following operating points were compared:
1. final locked serving threshold: `0.1279`
2. historical Week 12 reference threshold: `0.0884`
3. post-hoc recall-first reference threshold (illustrative only)

### B.3 Results

#### Final locked threshold `0.1279`
- Precision = 0.8280
- Recall = 0.8105
- F1 = 0.8191
- TP = 77
- FP = 16
- FN = 18
- TN = 56,635
- Alerts/10k = 16.39
- Cost/tx = 0.006626

#### Historical threshold `0.0884`
- Precision = 0.7938
- Recall = 0.8105
- F1 = 0.8021
- TP = 77
- FP = 20
- FN = 18
- TN = 56,631
- Alerts/10k = 17.09
- Cost/tx = 0.006697

#### Post-hoc recall-first reference threshold
- Threshold = 0.000054
- Precision = 0.0118
- Recall = 0.9053
- F1 = 0.0233
- TP = 86
- FP = 7,186
- FN = 9
- TN = 49,465
- Alerts/10k = 1281.50
- Cost/tx = 0.129807

### B.4 Interpretation
The final locked threshold `0.1279` preserved fraud recall relative to the historical `0.0884` reference while reducing false positives, improving precision, and slightly reducing operational cost. The post-hoc recall-first operating point achieved recall above 0.90, but at the expense of an operationally unrealistic alert burden.

---

## Appendix C — API Validation and Edge-Case Evidence

### C.1 Successful endpoint checks
The following endpoint checks were completed successfully during Week 21:
- `GET /health`
- `GET /metadata`
- valid `POST /predict_by_id`
- valid `POST /predict`

The API also handled valid edge-value inputs correctly:
- very large `Amount`
- `Amount = 0`
- `Time = 0`

### C.2 Deterministic invalid-input rejection
The following malformed inputs were rejected deterministically:
- missing required field
- unexpected extra field
- negative `Amount`
- negative `Time`
- string instead of numeric

These checks confirm that the hardened serving layer enforces strict input validation and avoids silent fallback predictions.

### C.3 Remaining hardening issue
Two non-finite numeric payloads surfaced as `500 Internal Server Error` instead of returning a clean validation response:
- `NaN`
- `Infinity`

#### NaN case
Validation logic was triggered, but the custom validation error response failed during JSON serialization because the error payload contained `nan`.

#### Infinity case
The preprocessing layer rejected the input explicitly with:
`ValueError: Input contains NaN or infinite values.`

### C.4 Appendix interpretation
These findings indicate that the API is robust for valid requests and for most malformed inputs. However, one remaining serving-layer hardening issue exists for non-finite numeric payloads, and this should be treated as a future improvement point rather than a model-quality issue.

---

## Appendix D — Peer Testing Materials
Week 21 also prepared external usability-testing materials:
- `reports/week21_user_testing/peer_feedback_log.csv`
- `reports/week21_user_testing/peer_testing_notes.md`

At the time of writing, peer feedback collection had been initiated and can be appended once the tester response is received.
