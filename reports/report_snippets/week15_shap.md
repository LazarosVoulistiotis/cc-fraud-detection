```markdown
## 4.X Model Explainability (SHAP)

To ensure the selected fraud detection model is transparent and auditable, SHAP (SHapley Additive exPlanations) was applied to the final XGBoost classifier. Since the dataset contains anonymised PCA features (V1–V28) and engineered features (e.g., hour_sin/hour_cos and Amount_log1p), explainability is essential to understand the model’s decision logic in a production-like setting.

### 4.X.1 SHAP setup
SHAP values were computed using **TreeExplainer**, which is efficient for tree-based models (XGBoost). To keep computation practical, global explanations were estimated on a random sample of 10,000 test rows with a background of 1,000 train rows. The operating decision threshold followed the Week 12 cost-based policy selected on the validation set (cost_fp=1, cost_fn=20), resulting in a threshold of **0.0884**, which was applied to the locked test set.

### 4.X.2 Global drivers
Figure X (beeswarm) and Figure Y (mean |SHAP| bar) show that the strongest global drivers are the PCA components. The top features by mean absolute SHAP include **V4 (0.675)**, **V14 (0.599)**, **V8 (0.495)**, and **V12 (0.369)**, followed by V15 and V11. This indicates that the model primarily relies on latent patterns captured in the PCA space, which is expected for anonymised fraud datasets.

**Figures:**
- `reports/figures/week15/shap_summary_beeswarm.png`
- `reports/figures/week15/shap_mean_abs_bar.png`

### 4.X.3 Dependence analysis
Two dependence plots were generated for the top features:
- **V4 dependence plot (colored by V12):** demonstrates non-linear threshold effects and feature interaction.
- **V14 dependence plot (colored by V11):** indicates threshold-like behaviour consistent with boosted trees and interaction with V11.

**Figures:**
- `reports/figures/week15/shap_dependence_V4.png`
- `reports/figures/week15/shap_dependence_V14.png`

### 4.X.4 Local case study explanations
To demonstrate decision-making at the transaction level, three representative examples from the test set were explained using SHAP waterfall plots:
- **True Positive (fraud detected):** index 18427, predicted probability ≈ 0.999998
- **True Negative (legitimate):** index 49260, predicted probability ≈ 8.33e-08
- **Borderline case (near threshold):** index 53293, predicted probability ≈ 0.088737 (threshold=0.0884)

These examples show how a small set of dominant features can strongly push the decision towards fraud or non-fraud, while borderline cases exhibit competing contributions that place the prediction close to the operating threshold.

**Figures:**
- `reports/figures/week15/shap_waterfall_true_positive.png`
- `reports/figures/week15/shap_waterfall_true_negative.png`
- `reports/figures/week15/shap_waterfall_borderline.png`