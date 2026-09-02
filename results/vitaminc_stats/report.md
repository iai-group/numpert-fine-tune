# VitaminC numerical subset: baseline vs adversarial fine-tuning

Instances evaluated by both models: 11887

## Per-model summary

| Model | n | Acc (all) | Acc (parsed) | Macro-F1 | Parse fail |
|---|---|---|---|---|---|
| baseline | 11887 | 90.96 | 96.68 | 96.64 | 5.92 |
| finetuned | 11887 | 78.35 | 78.35 | 77.63 | 0.00 |

## Paired comparison (parse failures counted as errors)

| Scope | n | Baseline | Fine-tuned | Delta (pp) | 95% CI | McNemar p |
|---|---|---|---|---|---|---|
| all instances | 11887 | 90.96 | 78.35 | -12.60 | [-13.42, -11.77] | 3.63e-192 |
| gold = supports | 6650 | 90.45 | 86.09 | -4.36 | [-5.32, -3.40] | 5.69e-19 |
| gold = refutes | 5237 | 91.60 | 68.53 | -23.07 | [-24.46, -21.71] | 2.12e-217 |
| contrastive pairs (both correct) | 5227 | 84.25 | 54.45 | -29.81 | [-31.36, -28.24] | 3.45e-271 |

Pair accuracy requires the model to label BOTH snapshots of a revision correctly, so it is the natural analogue of the paper's label-flipping evaluation.

## Pair accuracy by size of the numerical edit in the evidence

`refutes` = wrongly refuted the supporting snapshot; `confirms` = wrongly confirmed the refuting snapshot. Bin assignment uses the smallest relative change between the two snapshots and is indicative only (see `edit_size`).

| Edit size | Pairs | Base pair acc | FT pair acc | Delta (pp) | 95% CI | Base refutes/confirms | FT refutes/confirms |
|---|---|---|---|---|---|---|---|
| (0, 10%] | 3887 | 85.05 | 53.87 | -31.18 | [-32.98, -29.33] | 7.8 / 8.4 | 8.2 / 38.4 |
| (10%, 25%] | 586 | 83.11 | 72.01 | -11.09 | [-14.85, -7.17] | 9.9 / 8.9 | 17.1 / 13.5 |
| (25%, 50%] | 292 | 80.82 | 54.11 | -26.71 | [-33.56, -19.86] | 15.1 / 6.5 | 41.1 / 7.2 |
| (50%, 100%] | 235 | 85.53 | 30.21 | -55.32 | [-62.13, -48.09] | 9.8 / 6.4 | 61.3 / 10.6 |
| > 100% | 138 | 78.26 | 34.78 | -43.48 | [-52.90, -34.06] | 13.8 / 9.4 | 56.5 / 9.4 |

Pairs excluded because no changed number could be aligned: 89.
