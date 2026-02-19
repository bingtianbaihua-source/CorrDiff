# Task 1.7 (R7): Expert predictor evaluation (R2) during training

This validation bundle exercises the `ExpertPredictorEvaluator` R2 computation on a
synthetic batch containing:
- `z_pi` latents for the minimal property set (10 props)
- true labels for a subset of properties (5/10)

Expected output:
- `outputs/expert_r2.json` contains all 10 property keys
- properties without labels are marked as `"N/A"`

