# Packaging verification

Checked on November 11, 2026, using Python 3.12.3 and the direct dependency
versions listed in `requirements-repro.txt`.

- All 47 archived evaluations passed seed, achievement, rate, mean and bootstrap
  interval consistency checks. All six checkpoint weight hashes matched.
- The baseline and all five main policies each completed two identical
  64-step trajectories after reset, with valid actions on real observations.
- Three full run-5 episodes on seeds 2000–2002 exactly reproduced the archived
  achievement vectors and episode step counts. This checks packaging fidelity;
  these are already-used validation seeds, not new generalization evidence.
- A 64-transition PPO update completed with memory features, mixed shaping,
  warm-started weights and late-state-reset code enabled. The small run verifies
  execution and saving, not curriculum effectiveness or buffer population.
- The run-5 memory checker on seed 4000 reported one feature mismatch over
  1,151 steps (worst absolute error 1.0). This is consistent with the documented
  planting-event discrepancy; it does not establish equivalence on every world.
- Python syntax, local Markdown links and project metadata were checked.
  The generated result figure was visually inspected.

This verification did not rerun the complete 300/600-episode studies or retrain
the historical experiment lineage.
