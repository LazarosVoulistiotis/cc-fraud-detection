# Week 21 — Report Subsection Structure

## 1. Final hold-out evaluation of the frozen champion model
This subsection should report the final locked-test evaluation of the frozen XGBoost champion model using the frozen threshold configuration.
Include:
- model artifact used
- test dataset used
- threshold policy and threshold
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- confusion matrix
- fraud alert rate
- cost per transaction (if reported)

## 2. Validation of the locked threshold policy against project goals
This subsection should distinguish clearly between:
- the initial recall-first research goal
- the final business-aware serving goal

Include:
- comparison between threshold `0.1279` and historical threshold `0.0884`
- post-hoc reference threshold for recall >= 0.90 (explicitly not used for policy selection)
- interpretation of the operational trade-off

## 3. API robustness and edge-case testing
This subsection should summarize the final API-level verification results.
Include:
- successful checks for `/health`, `/metadata`, `/predict`, `/predict_by_id`
- deterministic rejection of missing / extra / invalid inputs
- note that `NaN` and `Infinity` surfaced as `500` and were recorded as final hardening issues

## 4. Peer usability testing and feedback
This subsection should summarize:
- who tested the system
- whether the tester understood the API/demo purpose
- whether `fraud_probability` and `predicted_label` were clear
- what confusion points were reported
- what small improvement actions were taken or proposed

## 5. Final revisions and release readiness
This subsection should summarize the final Week 21 revision pass.
Include:
- documentation cleanup
- evidence organization
- screenshot planning
- endpoint wording finalization
- hardening notes
- final Week 21 closure statement

## Closing sentence
Week 21 does not introduce a new model or a new threshold policy.
Instead, it demonstrates that the frozen fraud detection system is reproducible, technically validated, operationally interpretable, and presentation-ready as a deployment-oriented academic project.
