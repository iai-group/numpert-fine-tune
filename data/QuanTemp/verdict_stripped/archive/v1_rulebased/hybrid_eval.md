# Hybrid (rules -> LLM long-tail) eval

- records: 501 (errors: 0)

| cohort | n | residual_rules | residual_hybrid | improvement | locate_recall | over_removal (n) |
|---|---|---|---|---|---|---|
| overall | 501 | 0.273 | 0.054 | 0.219 | 0.927 | 0.157 (159) |
| True | 167 | 0.24 | 0.054 | 0.186 | 0.925 | 0.196 (51) |
| False | 167 | 0.281 | 0.06 | 0.221 | 0.915 | 0.14 (50) |
| Conflicting | 167 | 0.299 | 0.048 | 0.251 | 0.94 | 0.138 (58) |

residual_* = fraction the audit still flags as containing a verdict after rules / after rules+LLM. improvement = rules - hybrid. over_removal = fraction of LLM-removed records the audit flags as having lost real evidence.