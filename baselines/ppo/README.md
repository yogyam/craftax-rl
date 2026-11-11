# Reference PPO baseline

The supplied reference policy is a NumPy MLP with two 256-unit tanh layers,
17 sampled action logits and a training-only value output. The recorded training
budget was 15.9M environment steps over 90 minutes on an 8-vCPU machine.

The reference notes report 13.2 mean achievements over 100 external evaluation
episodes, compared with 12.3 for argmax actions, 2.1 for uniform random actions,
and 0 for no-op. Those external episode logs are unavailable here. The saved
local baseline evaluation covers seeds 1000–1099 and yields 13.13 (95% bootstrap
CI 12.62–13.61); see `results/evaluations/baseline_1000.json`.

The trainer defaults to 128 environments and a 40-minute budget; those defaults
do not reproduce the supplied checkpoint's training budget. Explicitly pass
`--num-envs 256 --budget-min 90` for the documented reference training recipe.

From the repository root, after installing the project:

```bash
python scripts/evaluate.py baselines/ppo --episodes 100 --seed-start 1000 --workers 4
python baselines/ppo/train.py --num-envs 256 --budget-min 90 --out checkpoints/baseline/weights.npz
```

The reference PPO implementation and checkpoint came from the original starter
code. The experiments extend that implementation; the baseline is not a new
algorithm introduced by this project.
