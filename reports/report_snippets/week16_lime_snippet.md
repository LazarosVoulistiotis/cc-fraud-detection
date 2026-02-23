# Week 16 — LIME (Local Explainability) + Comparison with SHAP

> **Goal:** Provide *on-demand* local explanations using a **model-agnostic** method (LIME), then compare against **SHAP** (Week 15) and document limitations for an industry-ready narrative.

---

## Setup (aligned with Week 15)

- **Model:** XGBoost champion (`models/xgb_week8.joblib`)
- **Operating threshold (VAL-selected, cost-policy):** `thr = 0.0884`
- **Cases (same as SHAP):**
  - **True Positive:** `idx = 18427` (y=1, p_fraud≈0.999998)
  - **True Negative:** `idx = 49260` (y=0, p_fraud≈0.0)
  - **Borderline:** `idx = 53293` (y=0, p_fraud≈0.088737, `thr=0.0884` → borderline FP by policy)
- **LIME parameters:** `num_samples = 5000`, `seed = 42`, `num_features = 10` (reproducible run)

---

## Step 3 — LIME vs SHAP comparison (case-by-case)

### Case A — True Positive (idx=18427, true=1, pred=1, p_fraud≈0.999998)

**LIME top drivers (towards fraud):**
- V14, V12, V4, V3, V10 …
- Engineered signals also appear as supportive terms: `hour_sin` / `hour_cos`.

**SHAP Week 15 (TP):**
- Core drivers already identified: **V14 / V12 / V10 / V4**.

✅ **Agreement on top drivers:** **Yes.** Same “core” fraud pattern appears in both SHAP and LIME (V14, V12, V4, V10).  
✅ **Agreement on sign (qualitative):**
- SHAP provides clear sign semantics (push up/down the fraud score).
- LIME provides *local* positive/negative weights for the fraud class; here the direction is consistent.

**Interpretation:** This is a *clean fraud* where both tools strongly agree. LIME additionally surfacing `hour_sin/hour_cos` suggests engineered time features can provide context in specific patterns, even if they are not global drivers.

---

### Case B — True Negative (idx=49260, true=0, pred=0, p_fraud≈0.0)

**LIME:**
- Dominantly **negative** contributions for the fraud class (red bars), with a few small positive terms.

✅ **Agreement:** **Yes.** Both methods indicate a very confident legitimate transaction (p≈0).  
⚠️ **Sign note (avoid confusion):** The plot is *“Local explanation for class fraud”*, so:
- **Red** terms reduce the fraud score,
- **Green** terms increase the fraud score.  
The net effect keeps p_fraud near zero → stable decision.

**Interpretation:** Strong anti-fraud signals outweigh any weak pro-fraud terms; these cases are typically more stable under LIME.

---

### Case C — Borderline (idx=53293, true=0, pred=1, p_fraud≈0.088737, thr=0.0884)

**LIME top drivers (towards fraud):**
- Again: V14, V12, V4, then V11/V3/V19/V10 …
- `Amount > 77.90` appears as a small push in this local explanation.

✅ **Agreement on top drivers:** **Yes.** Borderline resembles the TP pattern (V14/V12/V4/V10), but with weaker local evidence.  
✅/⚠️ **Agreement on sign:** Both indicate fraud-like pushes, but the sample is **threshold-sensitive**:
- The classification flips because it is just above the operational threshold.

**When disagreement is most likely:**
- **Non-linear regions / interactions** (boosted trees can have sharp interaction effects).
- **Borderline cases near the threshold**, where LIME’s local-linear surrogate can shift with sampling/seed.

**Business interpretation:** This borderline FP is an ideal **manual review** or **soft-action** candidate (e.g., step-up verification), since the decision is policy-driven rather than a high-confidence fraud.

---

## Overall conclusion (1 paragraph)

> Across the three shared case studies (TP/TN/borderline), LIME and SHAP show strong qualitative agreement on the core drivers of the XGBoost champion model, particularly for the clear fraud pattern (V14/V12/V4/V10). SHAP provides a more principled attribution with consistent sign semantics, while LIME offers an on-demand, model-agnostic local surrogate explanation expressed as human-readable rules (bins). Discrepancies are most likely near the operating threshold (borderline cases), where local linear approximations can be sensitive to sampling and non-linear interactions.

---

## Step 4 — LIME limitations (report-ready)

### (i) Stability / Reproducibility
LIME is stochastic. Changing **seed** or **num_samples** can change:
- the ranking of top terms,
- the estimated weights,
- especially for **borderline** samples.

**Good practice:** keep `seed=42` as the “main run”; optionally add 1–2 extra runs (e.g., seed=7 and seed=99) and note that ranking shifts are more visible near the threshold.

### (ii) Sensitivity to sampling / local surrogate bias
LIME generates synthetic perturbations and fits a **locally-weighted linear model** around the sample.
- Boosted trees often have **non-linear decision boundaries** and interactions.
- Therefore, LIME may approximate poorly in curved/interaction-heavy regions.

### (iii) Runtime / operational constraints
Each explanation requires sampling → runtime cost.
- In production, LIME should be used for a **small number** of analyst queries (on-demand).
- For systematic audit and consistent attributions, **SHAP** is the preferred method (especially for tree models).

### (iv) Practical takeaway (industry relevance)
> Use **SHAP** as the audit-grade explainability method (global + consistent local attributions), and **LIME** as an analyst-facing on-demand explanation tool for individual alerts.

---
