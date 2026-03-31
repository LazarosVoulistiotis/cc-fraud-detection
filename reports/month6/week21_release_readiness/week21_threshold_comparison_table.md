# Table — Week 21 Threshold Comparison

**Table X. Post-hoc comparison of threshold operating points on the locked test set (Week 21)**

| Threshold / Policy | Precision | Recall | F1 | TP | FP | FN | TN | Alerts/10k | Cost/tx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.1279` — final locked serving threshold | 0.8280 | 0.8105 | 0.8191 | 77 | 16 | 18 | 56,635 | 16.39 | 0.006626 |
| `0.0884` — historical reference threshold | 0.7938 | 0.8105 | 0.8021 | 77 | 20 | 18 | 56,631 | 17.09 | 0.006697 |
| `0.000054` — post-hoc recall-first reference | 0.0118 | 0.9053 | 0.0233 | 86 | 7,186 | 9 | 49,465 | 1281.50 | 0.129807 |

**Interpretation:**  
The final locked threshold `0.1279` preserved fraud recall relative to the historical `0.0884` reference while reducing false positives, improving precision, and slightly lowering expected cost. A recall-first operating point could exceed 0.90 recall, but only at the expense of an operationally unrealistic alert burden.
