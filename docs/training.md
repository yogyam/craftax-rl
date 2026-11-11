# Training recipes and lineage

Run commands from the repository root after installation. These are recipes for
the documented configurations, not promises of bitwise recreation. The runs
used wall-clock schedules, some were paused and restarted, and only weights
were checkpointed. CPU speed and evaluation overhead change the number of steps
completed within a time budget. The saved main-run checkpoints are the exact
weight files from the archived policy snapshots.

The trainer retains the reference defaults (2 × 256, 128 environments, λ=0.95),
so the larger configurations are explicit below. The recorded experiments ran
one JAX training process at a time on a laptop CPU. No accelerator is required.

## Run 1: larger PPO

```bash
python scripts/train.py --num-envs 1024 --width 512 --layers 3 \
  --budget-min 600 --lr 3e-4 --lam 0.9 --anneal-lr \
  --keep-ckpts --out checkpoints/run1/weights.npz
```

The recorded run completed about 277M steps. The supplied policy snapshot in
`policies/run1_ppo` can be evaluated without retraining.

## Run 2: broad upkeep shaping

```bash
python scripts/train.py --num-envs 1024 --width 512 --layers 3 \
  --budget-min 180 --lr 1e-4 --lam 0.9 --anneal-lr \
  --init-from policies/run1_ppo/weights.npz --shape-needs 0.1 \
  --keep-ckpts --out checkpoints/run2/weights.npz
```

Recorded budget: about 77M steps. This trains the network; the archived
`policies/run2_shaped/policy.py` adds the gated routines used for its main result.
The archived plain-network ablation used the same shaped weights without that
wrapper. Its JSON label is an experiment identifier, not a policy folder.

## Intermediate deficit-only fine-tune

```bash
python scripts/train.py --num-envs 1024 --width 512 --layers 3 \
  --budget-min 720 --lr 1e-4 --lam 0.9 --anneal-lr \
  --init-from policies/run2_shaped/weights.npz \
  --shape-needs 0.1 --shape-deficit 4 \
  --keep-ckpts --out checkpoints/deficit/weights.npz
```

The historical intermediate run accumulated about 253M steps across interrupted
segments. Its exact checkpoint is not published here. To follow this lineage,
produce that checkpoint locally before the next command. Starting run 3 directly
from the run 2 snapshot skips this phase and changes the experiment.

## Run 3: mixed shaping

```bash
python scripts/train.py --num-envs 1024 --width 512 --layers 3 \
  --budget-min 600 --lr 1e-4 --lam 0.9 --anneal-lr \
  --init-from checkpoints/deficit/weights.npz \
  --shape-needs 0.1 --deficit-drink 5 --deficit-food 5 --deficit-energy 99 \
  --keep-ckpts --out checkpoints/run3/weights.npz
```

Recorded total: about 192M steps across three segments. The main archived policy
uses the plant routine and lava mask. The full wrapper is an ablation, not the
default for `policies/run3_selective`.

## Run 4: memory features

```bash
python scripts/train.py --num-envs 1024 --width 512 --layers 3 \
  --budget-min 600 --lr 6e-5 --lam 0.9 --anneal-lr \
  --init-from policies/run3_selective/weights.npz --plant-feats \
  --shape-needs 0.1 --deficit-drink 5 --deficit-food 5 --deficit-energy 99 \
  --keep-ckpts --out checkpoints/run4/weights.npz
```

Recorded total: about 245M steps. With `--plant-feats`, the trainer automatically
zero-pads the first-layer kernel when warm-starting a 1,345-input checkpoint.
Inference must supply the matching 1,350-dimensional input. Check the tracking
on real trajectories with:

```bash
python scripts/check_memory.py policies/run4_memory --seeds 4000 4001 4002
```

The checker reports mismatches; it does not assert perfect equivalence. A
planting-step discrepancy and tracking drift can create a training/inference gap.

## Run 5: late-game resets

```bash
python scripts/train.py --num-envs 1024 --width 512 --layers 3 \
  --budget-min 600 --lr 6e-5 --lam 0.9 --anneal-lr \
  --init-from policies/run4_memory/weights.npz --plant-feats \
  --shape-needs 0.1 --deficit-drink 5 --deficit-food 5 --deficit-energy 99 \
  --late-resets 0.3 --late-buffer 256 \
  --keep-ckpts --out checkpoints/run5/weights.npz
```

The recorded run accumulated about 187M steps with a pause. The original notes
do not preserve every launch argument for this phase; the command above is a
working continuation recipe using the preceding memory configuration, not a
verbatim historical invocation. The bank captures at most one qualifying state
per vectorized step. The trigger is the transition to owning an iron sword while
also holding an iron pickaxe; obtaining the pickaxe after the sword does not
trigger capture in this implementation.

Evaluation always starts from fresh worlds; saved-state resets are a training
intervention. The bank, RNG, optimizer and environment states are not saved in
the weight file, so warm-starting does not resume those states.

## Evaluating and diagnosing

```bash
python scripts/evaluate.py policies/run4_memory \
  --episodes 300 --seed-start 4000 --workers 4 --out local-results/run4_4000.json
python scripts/evaluate.py policies/run5_curriculum \
  --episodes 300 --seed-start 4000 --workers 4 --out local-results/run5_4000.json
python scripts/diagnose.py policies/run5_curriculum \
  --episodes 20 --seed-start 4000 --workers 2
python scripts/trace.py policies/run2_shaped --seed 4000 --max-steps 1000
```

Use identical seed ranges for policy comparisons. The death classifier is a
heuristic based on the final state transition, not an authoritative cause-of-
death field supplied by the game. For new checkpoints, copy a compatible policy
script into a new folder beside its `weights.npz`; do not overwrite the archived
main-run snapshots when experimenting.
