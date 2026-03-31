# Week 21 — Endpoint Wording

## `/predict`
`POST /predict` is the primary inference endpoint of the fraud detection API.

It accepts a raw canonical transaction payload containing:
- `Time`
- `V1` to `V28`
- `Amount`

At request time, the API:
1. validates the payload,
2. applies deterministic preprocessing,
3. derives engineered features internally,
4. aligns the input to the frozen serving schema,
5. scores the frozen XGBoost champion model,
6. applies the locked threshold policy,
7. returns the fraud probability and final decision label.

### Short report wording
`POST /predict` simulates real inference for a new transaction by accepting raw transaction features and returning a thresholded fraud decision.

---

## `/predict_by_id`
`POST /predict_by_id` is a demo-oriented endpoint for deterministic testing and presentation.

It accepts:
- `row_id`

The API uses `row_id` to retrieve a prepared transaction from the demo/test dataset, then runs the same preprocessing, scoring, and thresholding pipeline as the main inference endpoint.

The response also includes:
- `row_id`
- `true_label`

### Short report wording
`POST /predict_by_id` is a reproducibility-friendly demo endpoint that allows the system to score a known test transaction by identifier, making live demonstrations and validation easier.

---

## Practical explanation
- Use `/predict` when you want to score a new raw transaction payload.
- Use `/predict_by_id` when you want a faster and more controlled demo using a known transaction from the prepared dataset.

## Interpretation note
- `fraud_probability` = the model score for the fraud class
- `predicted_label` = the final thresholded decision (`fraud` or `legit`)
