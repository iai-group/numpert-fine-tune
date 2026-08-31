# Hybrid (rules -> LLM long-tail) eval

- records: 501 (errors: 0)

| cohort | n | residual_rules | residual_hybrid | improvement | locate_recall | over_removal (n) |
|---|---|---|---|---|---|---|
| overall | 501 | 0.275 | 0.066 | 0.209 | 0.906 | 0.175 (183) |
| True | 167 | 0.24 | 0.048 | 0.192 | 0.925 | 0.218 (55) |
| False | 167 | 0.281 | 0.06 | 0.221 | 0.894 | 0.215 (65) |
| Conflicting | 167 | 0.305 | 0.09 | 0.215 | 0.902 | 0.095 (63) |

residual_* = fraction the audit still flags as containing a verdict after rules / after rules+LLM. improvement = rules - hybrid. over_removal = fraction of LLM-removed records the audit flags as having lost real evidence.