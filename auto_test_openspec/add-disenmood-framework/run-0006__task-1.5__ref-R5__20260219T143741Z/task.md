Task 1.5 (R5): training-free branch energy guidance with freezing.

This bundle runs a synthetic guidance loop over per-property latents `z_pi[k]`:
- `target` branches get gradient-based energy updates each step
- `frozen` branches remain unchanged across steps

Outputs:
- outputs/guided_latents.npz
- outputs/guidance_stats.json
