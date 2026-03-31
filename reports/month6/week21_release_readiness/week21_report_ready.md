## Week 21 — Final Model Validation, API Stress Testing, and Release Readiness

Week 21 served as the final validation and release-readiness phase of the project. No additional model training, threshold reselection, or serving-policy changes were introduced at this stage. Instead, the objective was to verify that the frozen fraud detection system remained reproducible, technically validated, operationally interpretable, and suitable for final presentation as a deployment-oriented academic project.

### 1. Final hold-out evaluation of the frozen champion model

The frozen XGBoost champion model was evaluated on the locked hold-out test set using the final serving configuration. The evaluation used the frozen artifact `models/xgb_final.joblib`, the engineered-schema test split `data/data_interim/splits_week8/test.csv`, and the locked threshold policy `precision_constraint_p80` with threshold `0.1279`.

The final hold-out results reproduced the expected locked-test outcome exactly. On the test set of 56,746 transactions, including 95 fraud cases, the system achieved a ROC-AUC of 0.96995 and a PR-AUC of 0.81713. At the frozen threshold, the model achieved Precision = 0.82796, Recall = 0.81053, and F1-score = 0.81915. The confusion matrix was TN = 56,635, FP = 16, FN = 18, TP = 77.

From an operational perspective, this corresponded to a fraud alert rate of 0.001639, or 16.39 alerts per 10,000 transactions, with an expected cost per transaction of 0.006626 under the project’s illustrative cost setting of FP = 1 and FN = 20. This exact reproduction of the locked test result confirms that the frozen model artifact, frozen schema, and frozen threshold configuration are aligned and reproducible.

### 2. Validation of the locked threshold policy against project goals

A post-hoc sensitivity analysis was conducted on the locked test predictions in order to compare the final serving threshold with historical and illustrative alternatives. This analysis was reported strictly as comparison only and was not used to re-select or alter the production-facing threshold policy.

Three operating points were examined: the final locked threshold `0.1279`, the historical Week 12 reference threshold `0.0884`, and a post-hoc recall-first reference threshold chosen only to illustrate what would be required to push recall above 0.90.

At the final locked threshold `0.1279`, the model achieved Precision = 0.8280, Recall = 0.8105, F1 = 0.8191, with TP = 77, FP = 16, FN = 18, and TN = 56,635. At the historical threshold `0.0884`, the model achieved Precision = 0.7938, Recall = 0.8105, and F1 = 0.8021, with TP = 77, FP = 20, FN = 18, and TN = 56,631. Therefore, the historical threshold did not improve fraud recall on the locked test set, but it did increase false positives, reduce precision, and slightly worsen the operational cost profile.

A post-hoc recall-first reference threshold was also examined for sensitivity purposes. That operating point pushed recall to 0.9053, but precision collapsed to 0.0118 and false positives rose dramatically to 7,186, corresponding to 1,281.50 alerts per 10,000 transactions. This operating point is therefore not operationally viable and was reported only as a stress-test comparison.

These results show that the frozen serving threshold `0.1279` offers the best operational balance for the deployed system: it preserves fraud capture relative to the historical reference while lowering false positives, improving precision, and slightly reducing expected cost.

### 3. Goal assessment: research aspiration versus serving objective

Week 21 distinguished explicitly between two different project goals.

The first was the initial research-oriented goal defined earlier in the project: a recall-first target of Recall ≥ 0.90 with acceptable Precision. Under the final serving threshold, this aspiration was not met, because fraud recall remained at approximately 0.81. Although a post-hoc threshold could increase recall above 0.90, doing so caused precision to collapse and generated an unsustainable alert burden.

The second was the final serving-oriented goal that emerged as the project matured: use a frozen validation-selected threshold that maintains strong operational precision, reduces false positives, keeps analyst workload manageable, and avoids unnecessary customer friction. Under this criterion, the frozen threshold `0.1279` clearly outperformed the historical `0.0884` reference and therefore better satisfied the final business-aware objective of the deployed system.

This distinction is important. The fact that the system did not preserve the early recall-first aspiration at serving time should not be treated as a weakness. Rather, it reflects a mature and explicitly justified transition from an exploratory research target to a more realistic production-inspired operating policy.

### 4. API robustness and edge-case testing

A final API-level robustness pass was performed on the frozen FastAPI serving layer. The service behaved correctly for the main valid endpoints: `GET /health`, `GET /metadata`, valid `POST /predict_by_id`, and valid `POST /predict`. It also handled valid edge-value inputs correctly, including `Amount = 0` and very large transaction amounts.

The API further demonstrated deterministic rejection for most malformed inputs. Missing required fields, unexpected extra fields, negative `Amount`, negative `Time`, and invalid numeric strings were all rejected cleanly through validation, rather than silently producing predictions. This confirms that the serving layer enforces the frozen schema strictly and behaves in a way that is both technically safe and academically defensible.

However, Week 21 also identified one remaining hardening issue. Non-finite numeric values (`NaN` and `Infinity`) surfaced as `500 Internal Server Error` instead of returning a clean client-facing validation response. In the `NaN` case, validation logic was triggered, but the custom error handling path failed during JSON serialization because the error payload contained `nan`. In the `Infinity` case, the preprocessing layer explicitly rejected the input as invalid, but the final response still surfaced as a generic server error. This issue does not affect model quality or the frozen threshold policy; it is a serving-layer hardening gap in error handling for non-finite numeric payloads.

### 5. Peer usability testing and feedback

Week 21 also prepared a peer-testing workflow to gather external usability feedback on the live deployed demo. Supporting materials were created to record tester identity, scenario, issue observed, severity, and action taken, along with a structured note template containing four short usability questions: whether the tester understood what the demo does, whether `fraud_probability` and `predicted_label` were clear, whether anything was confusing, and whether the system looked complete enough for portfolio or viva presentation.

At the time of writing, peer feedback collection had been initiated but was still pending. The final tester response can therefore be appended to the report or included in the appendix once received.

### 6. Final revisions and release readiness

In addition to validation, Week 21 included a final revision pass focused on documentation quality, evidence organization, and presentation readiness. This included preparing an evidence manifest, documenting the `NaN` / `Infinity` hardening issue, finalizing the wording for the `/predict` and `/predict_by_id` endpoints, drafting the Week 21 report subsection structure, preparing a final decision memo, and updating the revision checklist.

As a result, Week 21 did not change the model, threshold policy, or serving configuration. Instead, it provided the final technical validation, threshold justification, API verification, documentation cleanup, and evidence organization required to present the project as a complete and deployment-oriented fraud detection system.

### Final Week 21 conclusion

Overall, Week 21 confirmed that the frozen fraud detection system is ready to be presented as a real-world-inspired academic machine learning deployment project. The artifact, schema, and threshold configuration were validated successfully; the locked hold-out metrics were reproduced exactly; the final threshold policy was justified against both research and operational goals; and the API was shown to be robust for valid requests and for most malformed inputs. The only remaining gap identified was a final hardening issue for non-finite numeric inputs, which was documented transparently as a serving-layer improvement point rather than a modelling issue.