# Week 21 — NaN / Infinity Hardening Note

## Summary
The Week 21 API robustness pass confirmed that the frozen FastAPI serving layer rejects most malformed inputs deterministically.
However, a specific hardening gap was identified for non-finite numeric values such as `NaN` and `Infinity`.

## Observed behavior
- `NaN` input did not return a clean `422` validation response.
- `Infinity` input did not return a clean `422` validation response.
- In both cases, the API returned `500 Internal Server Error`.

## Technical interpretation
The invalid values were detected as problematic, but the final client-facing response was not handled gracefully.

### NaN case
The request triggered validation logic, but the custom validation error response path attempted to serialize a payload containing `nan`, which caused JSON rendering to fail.

### Infinity case
The preprocessing layer explicitly rejected the input with:
`ValueError: Input contains NaN or infinite values.`

## Conclusion
This is not a model-quality issue and does not affect the frozen model artifact or locked threshold policy.
It is a serving-layer hardening issue in error handling for non-finite numeric payloads.

## Recommended treatment in report
Report transparently that:
- the API is robust for normal and most invalid inputs,
- but `NaN` and `Infinity` currently surface as `500` instead of a clean validation response,
- and this is identified as a final production-hardening improvement point.

## Suggested future fix
Sanitize non-finite values before building JSON error responses, or normalize validation error payloads so that `nan` / `inf` never reach JSON serialization unchanged.
