# Week 21 — Final Validation, Robustness Review, and Release Readiness

Week 21 constituted the final validation and release-readiness phase of the project. At this stage, no further model training, threshold reselection, or serving-policy modification was introduced. Instead, the objective was to confirm that the frozen fraud detection system remained reproducible, technically sound, operationally interpretable, and suitable for final presentation as a deployment-oriented academic machine learning project.

## Final hold-out evaluation of the frozen champion model

The final evaluation was conducted using the frozen XGBoost champion artifact (`models/xgb_final.joblib`), the engineered-schema locked test split (`data/data_interim/splits_week8/test.csv`), and the final serving threshold policy `precision_constraint_p80` with threshold `0.1279`. This ensured that the Week 21 assessment matched the deployed system configuration exactly, without introducing any additional modelling or thresholding changes.

The locked hold-out results reproduced the expected frozen-system outcome exactly. On the test set of 56,746 transactions, including 95 fraud cases, the model achieved a ROC-AUC of 0.96995 and a PR-AUC of 0.81713. At the frozen serving threshold, Precision reached 0.82796, Recall reached 0.81053, and F1-score reached 0.81915. The corresponding confusion matrix was TN = 56,635, FP = 16, FN = 18, and TP = 77.

From an operational perspective, this corresponded to a fraud alert rate of 0.001639, or 16.39 alerts per 10,000 transactions, with an illustrative expected cost per transaction of 0.006626 under the cost setting FP = 1 and FN = 20. These findings confirm that the frozen model artifact, frozen serving schema, and locked threshold configuration are internally aligned and reproducible.

## Validation of the locked threshold policy against project goals

To support the final interpretation of the serving decision, Week 21 included a post-hoc sensitivity analysis comparing the frozen serving threshold with historical and illustrative alternatives. This analysis was reported strictly as comparison only and was not used to redefine the serving policy from the test set.

Three operating points were examined: the final locked threshold `0.1279`, the historical Week 12 reference threshold `0.0884`, and a post-hoc recall-first operating point selected only to illustrate the conditions required to push recall above 0.90.

At the final locked threshold `0.1279`, the model achieved Precision = 0.8280, Recall = 0.8105, and F1 = 0.8191, with 16 false positives. At the historical threshold `0.0884`, Precision decreased to 0.7938 while Recall remained unchanged at 0.8105, and false positives increased to 20. Therefore, the historical threshold did not improve fraud capture on the locked test set, but it did worsen the operational false-positive burden.

A recall-first post-hoc operating point increased Recall to 0.9053, but Precision collapsed to 0.0118 and false positives rose to 7,186, producing 1,281.50 alerts per 10,000 transactions. This operating point is clearly not viable for practical serving and was included only as a sensitivity illustration.

These results make the threshold interpretation straightforward. The final serving threshold `0.1279` preserves fraud recall relative to the historical reference while reducing false positives, improving precision, and slightly lowering expected cost. Accordingly, it remains the most suitable operational choice for the frozen deployed system.

## Goal assessment: research aspiration versus final serving objective

An important aspect of Week 21 was the distinction between two different project goals.

The first was the initial research-oriented aspiration defined earlier in the project: a recall-first objective targeting Recall ≥ 0.90 with acceptable Precision. Under the final serving configuration, this aspiration was not met, because fraud recall remained at approximately 0.81. While a lower threshold could indeed push recall above 0.90, the resulting operating point would be associated with an unsustainable explosion in false positives and a severe collapse in precision.

The second was the final serving-oriented objective that emerged as the project matured: preserve a frozen validation-selected threshold that maintains strong operational precision, limits false positives, reduces analyst workload, and lowers avoidable customer friction. Under this criterion, the frozen threshold `0.1279` clearly outperformed the historical `0.0884` reference and therefore better satisfied the final business-aware objective of the deployed system.

This distinction is critical for correct interpretation. The fact that the final serving threshold does not satisfy the original recall-first aspiration should not be interpreted as a weakness of the project. Rather, it reflects a deliberate and justified transition from an exploratory research target to a more realistic production-inspired operating policy.

## API robustness and edge-case testing

Week 21 also included a final API-level robustness review of the frozen FastAPI serving layer. The service behaved correctly for the main valid endpoint paths, including `GET /health`, `GET /metadata`, valid `POST /predict_by_id`, and valid `POST /predict`. In addition, valid edge-value inputs such as `Amount = 0`, `Time = 0`, and very large transaction amounts were processed successfully without breaking the inference path.

The API also demonstrated deterministic rejection for most malformed inputs. Missing required fields, unexpected extra fields, negative `Amount`, negative `Time`, and invalid numeric strings were all rejected cleanly through validation rather than producing silent fallback predictions. This confirms that the serving layer enforces strict schema validation and supports reproducible, controlled inference-time behavior.

However, the Week 21 stress tests identified one remaining serving-layer hardening issue. Non-finite numeric payloads (`NaN` and `Infinity`) surfaced as `500 Internal Server Error` rather than returning clean client-facing validation responses. In the `NaN` case, validation logic was triggered, but the custom validation error response failed during JSON serialization because the error payload contained `nan`. In the `Infinity` case, the preprocessing layer explicitly rejected the payload as invalid, but the final response still surfaced as a generic server error. This issue does not affect model quality, threshold selection, or the reproducibility of the frozen artifact; rather, it represents a final API hardening gap in the handling of non-finite numeric payloads.

## Peer usability testing and external feedback

To strengthen the final presentation narrative, Week 21 also prepared a peer-testing workflow for external usability feedback on the live deployed demo. Structured materials were prepared to record tester identity, scenario, issue observed, severity, and action taken, together with a short note template containing questions on user understanding, output clarity, confusion points, and perceived completeness of the system for portfolio or viva presentation.

At the time of writing, the peer-testing workflow had been initiated but the external tester response was still pending. This does not affect the technical closure of Week 21, but the resulting feedback can be incorporated later as supplementary usability evidence.

## Final revisions and release readiness

The final portion of Week 21 focused on revision and presentation readiness rather than further experimentation. This included organizing evidence files, drafting a release-readiness checklist, documenting the `NaN` / `Infinity` hardening issue, finalizing the wording for `/predict` and `/predict_by_id`, and preparing a clear subsection structure for the final report and appendix.

As a result, Week 21 did not alter the model, threshold, or deployment configuration. Instead, it consolidated the final evidence required to present the system as a coherent end-to-end fraud detection solution with a frozen model, hardened API, Dockerized runtime, live cloud deployment, and transparent operational interpretation.

## Final Week 21 conclusion

Overall, Week 21 confirmed that the frozen fraud detection system is ready to be presented as a real-world-inspired academic machine learning deployment project. The frozen artifact, serving schema, and threshold configuration were validated successfully; the locked hold-out metrics were reproduced exactly; the final threshold policy was justified against both research and operational objectives; and the API was shown to be robust for valid requests and for most malformed inputs. The only remaining issue identified was a final hardening gap for non-finite numeric inputs, which was documented transparently as a serving-layer improvement point rather than a modelling issue.
