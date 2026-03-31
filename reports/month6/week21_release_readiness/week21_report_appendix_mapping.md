# Week 21 — Main Report vs Appendix Mapping

## Main Report
The following Week 21 material should be summarized in the main report:

### 1. Final hold-out evaluation of the frozen champion model
Use:
- final locked-test metrics
- confusion matrix
- short operational reading

### 2. Validation of the locked threshold policy against project goals
Use:
- comparison between `0.1279` and `0.0884`
- short mention of the post-hoc recall-first reference
- clear interpretation of operational trade-off

### 3. API robustness and edge-case testing
Use:
- short summary of successful endpoint checks
- short summary of deterministic invalid-input rejection
- explicit note that `NaN` / `Infinity` surfaced as `500`

### 4. Peer usability testing and feedback
Use:
- short note that peer testing was initiated
- add the final feedback summary when available

### 5. Final revisions and release readiness
Use:
- one short paragraph on Week 21 documentation cleanup
- one short paragraph on evidence organization and presentation readiness

---

## Appendix
The following Week 21 material should be moved to the appendix:

### Appendix A — Freeze and hold-out evidence
- freeze summary
- holdout metrics JSON
- scored test predictions reference

### Appendix B — Threshold sensitivity evidence
- threshold comparison JSON / CSV
- goal assessment note

### Appendix C — API validation evidence
- health response
- metadata response
- successful `/predict_by_id`
- successful `/predict`
- missing field validation output
- extra field validation output
- negative `Amount` validation output
- negative `Time` validation output
- string parsing validation output
- `NaN` 500 response
- `Infinity` 500 response

### Appendix D — Release-readiness notes
- evidence manifest
- endpoint wording
- NaN / Infinity hardening note
- Week 21 decision memo
- revision checklist

### Appendix E — Peer testing
- peer testing notes
- peer feedback log
- tester response summary (when received)

---

## Practical rule
Main report = short, decision-oriented, readable.
Appendix = raw evidence, outputs, detailed error responses, and technical support material.
