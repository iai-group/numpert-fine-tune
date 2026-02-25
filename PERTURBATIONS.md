# Numerical Claim Perturbations

This document describes the perturbation types applied to numerical claims in the dataset, the naming conventions used, and how the output files are structured.

---

## Perturbation Types

Perturbations are categorised into three types based on their relationship to the original claim's truth value.

### `perturb_type1` — Structure-Preserving
The claim is rewritten to retain the same numerical structure and meaning as the original, but expressed differently (e.g. converting digits to words). The perturbed claim should remain semantically equivalent to the original.

**Modules:** `numeration`, `approximation`, `range`

---

### `perturb_type2` — Value-Modifying
The numerical value in the claim is altered in a meaningful, controlled way. The perturbed claim is numerically different from the original and therefore has a different truth value.

**Modules:** `numeration`, `approximation`, `range`, `negative`

| Module | Modification |
|---|---|
| `numeration` | Scales the number by a fixed factor (e.g. ×1.1) |
| `approximation` | Replaces exact values with approximate expressions (e.g. "about 50") |
| `range` | Replaces exact values with a range (e.g. "between 40 and 60") |
| `negative` | Negates percentage values (e.g. 50% → -50%) |

---

### `perturb_type3` — Obfuscating
The numerical value is obscured or destroyed entirely. There is no truth-preserving counterpart — the original value is gone. These perturbations are grouped under the `obfuscation` module and come in three variants:

| Key | Description |
|---|---|
| `perturb_type3_random_{label}` | Digits replaced with a randomly generated number |
| `perturb_type3_masked_{label}` | Digits replaced with `#` symbols |
| `perturb_type3_removed_{label}` | Numerical entities removed from the claim entirely |

---

## File Structure

Output files are split by **perturbation module** and **gold label** of the original claim. Each file contains all perturbation type variants for that module as keys within each entry.

```
data/perturbed_data/
└── train_split/
    ├── approximation_true.json
    ├── approximation_false.json
    ├── numeration_true.json
    ├── numeration_false.json
    ├── range_true.json
    ├── range_false.json
    ├── negative_true.json
    ├── negative_false.json
    ├── obfuscation_true.json
    └── obfuscation_false.json
```

The `_true` / `_false` suffix refers to the **gold label of the original claim**, not whether the perturbation preserved the truth value.

---

## Entry Format

Each entry retains the original metadata keys (defined in `INCLUDED_KEYS`) alongside one or more perturbation keys.

### Example — `numeration_true.json`
```json
{
    "id": "abc123",
    "claim": "The rate increased by 50 percent.",
    "normalized_claim": "The rate increased by 50 percent.",
    "label": true,
    "perturb_type1_True": "The rate increased by fifty percent.",
    "perturb_type2_True": "The rate increased by fifty-five percent."
}
```

### Example — `obfuscation_false.json`
```json
{
    "id": "def456",
    "claim": "GDP grew by 3.2% in 2020.",
    "normalized_claim": "GDP grew by 3.2% in 2020.",
    "label": false,
    "perturb_type3_random_False": "GDP grew by 7.8% in 2020.",
    "perturb_type3_masked_False": "GDP grew by #.#% in ####.",
    "perturb_type3_removed_False": "GDP grew by % in."
}
```

Not all perturbation keys are guaranteed to be present in every entry — a key is only added if the perturbation produced a change from the original claim.

---

## Summary Table

| Module | File prefix | Type 1 | Type 2 | Type 3 |
|---|---|---|---|---|
| Numeration | `numeration` | ✓ | ✓ | — |
| Approximation | `approximation` | ✓ | ✓ | — |
| Range | `range` | ✓ | ✓ | — |
| Negative | `negative` | — | ✓ | — |
| Obfuscation | `obfuscation` | — | — | ✓ (×3 variants) |
