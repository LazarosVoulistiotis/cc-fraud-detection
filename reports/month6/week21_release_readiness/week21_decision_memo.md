# Week 21 — Final Decision Memo

## Decision
The Week 21 review confirms that the frozen fraud detection system is ready to be presented as a deployment-oriented academic project.

## What was verified
- The frozen artifact, serving schema, and threshold configuration were verified successfully.
- The final hold-out evaluation reproduced the locked test results of the champion model.
- Threshold validation was performed as post-hoc sensitivity analysis only, without re-tuning.
- The final serving threshold `0.1279` was confirmed as a better operational choice than the historical `0.0884` reference.
- The API serving layer was validated for successful inference and for deterministic rejection of most malformed inputs.

## Important interpretation
The final serving threshold does not satisfy the original recall-first aspiration of recall greater than or equal to 0.90.
However, it does satisfy the final business-aware serving objective by maintaining strong precision, controlled false positives, and lower operational burden.

## Final API note
The Week 21 edge-case pass identified one remaining hardening issue:
non-finite numeric values (`NaN` and `Infinity`) currently surface as `500 Internal Server Error` instead of a clean validation response.

## Outcome
No model retraining, no threshold reselection, and no serving-policy changes are introduced in Week 21.
Instead, Week 21 provides final validation, documentation cleanup, and release-readiness evidence for the frozen system.

## Final closure statement
Week 21 demonstrates that the final fraud detection system is reproducible, technically validated, operationally interpretable, and suitable for presentation as a real-world-inspired machine learning deployment project.
