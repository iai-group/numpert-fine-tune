# Hybrid (rules -> LLM long-tail) eval

- records: 501 (errors: 0)

| cohort | n | residual_rules | residual_hybrid | improvement | locate_recall | over_removal (n) |
|---|---|---|---|---|---|---|
| overall | 501 | 0.23 | 0.038 | 0.192 | 0.974 | 0.37 (184) |
| True | 167 | 0.198 | 0.042 | 0.156 | 0.939 | 0.463 (54) |
| False | 167 | 0.246 | 0.03 | 0.216 | 0.976 | 0.47 (66) |
| Conflicting | 167 | 0.246 | 0.042 | 0.204 | 1.0 | 0.188 (64) |

residual_* = fraction the audit still flags as containing a verdict after rules / after rules+LLM. improvement = rules - hybrid. over_removal = fraction of LLM-removed records the audit flags as having lost real evidence.