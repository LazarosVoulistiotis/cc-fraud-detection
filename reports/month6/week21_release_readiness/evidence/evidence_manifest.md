# Week 21 — Evidence Manifest

## Suggested screenshots / captures
- `/docs` landing page
- successful `GET /metadata`
- successful `POST /predict_by_id`
- successful `POST /predict`
- failed validation: missing required field
- failed validation: extra unexpected field
- failed validation: negative `Amount`
- failed validation: negative `Time`
- failed validation: string instead of numeric
- failed edge case: `NaN`
- failed edge case: `Infinity`

## Suggested appendix grouping
### Main report candidates
- `/docs`
- `/metadata`
- successful `/predict`
- successful `/predict_by_id`
- short summary of invalid input handling
- short note on NaN / Infinity hardening gap

### Appendix-only candidates
- missing field full JSON error
- extra field full JSON error
- negative `Amount` full JSON error
- negative `Time` full JSON error
- string numeric parsing error
- `NaN` 500 response
- `Infinity` 500 response

## Notes
Keep filenames clear and sequential.
Prefer a small number of high-value screenshots in the main report and move the rest to the appendix.
