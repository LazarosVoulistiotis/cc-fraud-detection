# Week 21 — Goal Assessment

## Goal A — Initial research goal
Initial early-project target:
- Recall (Fraud) ≥ 0.90
- Precision ≥ 0.10

Observed from Week 21 sensitivity analysis:
- Final locked threshold `0.1279`:
  - Precision = 0.8280
  - Recall = 0.8105
- Post-hoc recall-first reference:
  - Precision = 0.0118
  - Recall = 0.9053

Conclusion:
The final serving threshold does **not** satisfy the initial recall-first research aspiration, because recall remains below 0.90.
The post-hoc threshold that pushes recall above 0.90 causes precision collapse and an unsustainable number of false positives.

## Goal B — Final serving goal
Final system objective:
- frozen validation-selected serving threshold
- operationally acceptable precision
- fewer false positives
- lower analyst workload
- lower customer friction

Observed from Week 21:
- Final locked threshold `0.1279`:
  - Precision = 0.8280
  - Recall = 0.8105
  - FP = 16
  - Alerts/10k = 16.39
- Historical threshold `0.0884`:
  - Precision = 0.7938
  - Recall = 0.8105
  - FP = 20
  - Alerts/10k = 17.09

Conclusion:
The frozen threshold `0.1279` satisfies the final business-aware serving goal better than the historical `0.0884`, because it preserves fraud recall while reducing false positives, improving precision, and slightly reducing operational cost.

## Final Week 21 conclusion
The final serving threshold does not satisfy the original recall > 0.90 research aspiration.
However, it does satisfy the final operational serving policy that was locked on validation and carried into the frozen deployment stack.

This is not a weakness of the project.
It is a mature and explicitly justified business decision: the final system prioritizes a more practical precision–recall balance for serving, instead of maximizing recall at any operational cost.
