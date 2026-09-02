# Numerical robustness across two test sets

`VitaminC` probes naturally occurring evidence-side numerical revisions; `QuanTemp` probes synthetic claim-side perturbations. Pair accuracy requires both snapshots of one revision to be labelled correctly.

| Model | VitaminC acc | VitaminC macro-F1 | VitaminC pair acc | VitaminC parse fails | QuanTemp acc | QuanTemp macro-F1 |
|---|---|---|---|---|---|---|
| Base Qwen3-8B, zero-shot | 90.96 | 90.86 | 84.25 | 704 | — | — |
| QuanTemp-tuned | 78.35 | 77.63 | 54.45 | 0 | 97.52 | 94.62 |
| QuanTemp-tuned + threshold prompt | 85.16 | 84.85 | 69.16 | 0 | — | — |
| QuanTemp-tuned + structured CoT | 74.98 | 73.20 | 46.15 | 0 | — | — |
| QuanTemp-tuned + PAP | 76.18 | 76.00 | 53.74 | 0 | — | — |
| VitaminC-trained | 98.05 | 98.02 | 96.56 | 0 | 80.86 | 69.80 |
| Mixed QuanTemp + VitaminC | 98.25 | 98.23 | 97.15 | 0 | 98.20 | 96.10 |

Instances per test set: QuanTemp 6,175, VitaminC 11,887

Notes:
- Base Qwen3-8B, zero-shot: reasons with chain-of-thought; 5.9% parse failures
- QuanTemp-tuned + threshold prompt: prompt selected on the VitaminC dev split

## QuanTemp accuracy by perturbation type

| Model | original_normalized | perturb_type1 | perturb_type2 | perturb_type3_masked | perturb_type3_random |
|---|---|---|---|---|---|
| QuanTemp-tuned | 96.13 | 94.95 | 98.80 | 100.00 | 99.86 |
| VitaminC-trained | 79.29 | 72.69 | 92.26 | 57.88 | 95.12 |
| Mixed QuanTemp + VitaminC | 96.36 | 96.45 | 99.45 | 100.00 | 99.72 |
