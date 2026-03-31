# Week 21 — Word / Final Report Integration Notes

## Main report placement
Insert Week 21 in the main report as a structured section with the following subsections:

### 1. Final hold-out evaluation of the frozen champion model
Use:
- short paragraph introducing frozen evaluation setup
- one compact table with final metrics
- one short paragraph interpreting confusion matrix and operational reading

### 2. Validation of the locked threshold policy against project goals
Use:
- one comparison paragraph for `0.1279` vs `0.0884`
- one sentence on the post-hoc recall-first reference
- one concluding sentence explaining why `0.1279` remained the final serving threshold

### 3. API robustness and edge-case testing
Use:
- one short paragraph summarizing successful endpoint checks
- one short paragraph summarizing deterministic invalid-input rejection
- one short paragraph documenting the `NaN` / `Infinity` hardening issue

### 4. Peer usability testing and feedback
Use:
- a short placeholder paragraph for now
- later add tester name/type, summary of feedback, and any small improvement action

### 5. Final revisions and release readiness
Use:
- one short paragraph on Week 21 cleanup and evidence organization
- one short paragraph on final closure/readiness

---

## Suggested table for main report

### Table title
**Table X. Final locked-test evaluation of the frozen XGBoost champion model (Week 21)**

### Table contents
- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- TN
- FP
- FN
- TP
- Alerts per 10,000 transactions
- Cost per transaction

---

## Suggested figure candidates for main report

### Figure 1
**Figure X. Swagger /docs interface of the deployed fraud detection API**
Caption:
Screenshot of the live FastAPI Swagger interface used to test the deployed fraud detection service.

### Figure 2
**Figure X. Metadata response of the frozen serving system**
Caption:
Example metadata response showing the frozen model version, threshold policy, threshold value, and schema information exposed by the serving layer.

### Figure 3
**Figure X. Example successful prediction response**
Caption:
Example valid prediction response returned by the frozen API, including fraud probability and thresholded decision label.

### Figure 4 (optional)
**Figure X. Example validation error response**
Caption:
Example deterministic validation error returned when a malformed request is submitted to the API.

---

## Suggested appendix placement

### Appendix A
Freeze and hold-out evidence

### Appendix B
Threshold sensitivity comparison

### Appendix C
API validation outputs and edge-case responses

### Appendix D
Release-readiness notes

### Appendix E
Peer testing notes and feedback log

---

## Practical writing rule
Keep the main report decision-oriented and concise.
Move raw JSON outputs, long validation errors, and full screenshots with technical detail to the appendix.
