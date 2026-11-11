"""Check all policy snapshots against observations from the real environment.

This checks loading, episode reset, determinism and valid actions. It is not a
replacement for the full multi-seed evaluation.
"""

import importlib.util
import json
from pathlib import Path
import sys

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from craftax_rl.env import make_env


def main():
    env, params = make_env()
    reset = jax.jit(lambda key: env.reset(key, params))
    step = jax.jit(lambda key, state, action: env.step_env(key, state, action, params))
    policies = json.loads((ROOT / "results/checkpoints.json").read_text())
    for index, folder in enumerate(policies):
        spec = importlib.util.spec_from_file_location(f"policy_{index}", ROOT / folder / "policy.py")
        policy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy)
        actions = []
        for repeat in range(2):
            key = jax.random.PRNGKey(424242)
            obs, state = reset(key)
            assert obs.shape == (1345,)
            policy.reset()
            current = []
            for t in range(64):
                action = policy.act(np.asarray(obs, np.float32).copy())
                assert isinstance(action, (int, np.integer)) and 0 <= action < 17
                current.append(int(action))
                obs, state, _, done, _ = step(jax.random.fold_in(key, t), state, int(action))
                if bool(done):
                    break
            actions.append(current)
        assert actions[0] == actions[1], folder
        print(f"PASS {folder}: {len(actions[0])} valid steps; repeat after reset matched.")


if __name__ == "__main__":
    main()
