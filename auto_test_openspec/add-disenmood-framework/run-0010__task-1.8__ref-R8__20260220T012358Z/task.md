# Task 1.8 (R8) — Toy evaluation pipeline (CLI)

## How to run

macOS/Linux:

```bash
bash run.sh
```

Windows:

```bat
run.bat
```

## Inputs / Outputs

- Input config: `inputs/eval_config.json`
- Output report: `outputs/eval_report.json`

## Metrics computed

- `HV`: dominated hypervolume (maximization) for 4 objectives `[qed, sa, logp, -affinity]` vs `hv_reference`
- `Sparsity`: mean pairwise Euclidean distance over raw objective vectors `[qed, sa, logp, affinity]`
- `correlation_error`: Frobenius norm `||C_pred - C_true||_F`
- `intervention_result`: passthrough numeric summary from config (`property_varied`, `delta_target`, `delta_others_max`)
- `docking_scores`: mock docking scores list from config

## Pass/fail criteria (machine-decidable)

`run.sh`/`run.bat` MUST:

1. Produce `outputs/eval_report.json`
2. Verify JSON contains keys: `HV`, `Sparsity`, `correlation_error`, `intervention_result`, `docking_scores`

