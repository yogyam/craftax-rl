"""Verify the policy's reconstructed memory features match the true env state, step by step.
Usage: python scripts/check_memory.py policies/run4_memory --seeds 2000 2001 2002
AI-use note: drafted with Claude Code."""
import argparse, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser(); ap.add_argument("policy_dir"); ap.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
ap.add_argument("--max-steps", type=int, default=3000)
a = ap.parse_args()
import jax
from craftax_rl.env import PARAMS, make_env
env, _ = make_env(); reset = jax.jit(lambda k: env.reset(k, PARAMS))
@jax.jit
def step(key, t, state, action):
    o, st, _, d, _ = env.step_env(jax.random.fold_in(key, t), state, action, PARAMS); return o, st, d
sys.path.insert(0, os.path.abspath(a.policy_dir)); import policy as P

def true_feats(st):
    pos = np.asarray(st.player_position); gp = np.asarray(st.growing_plants_positions); gm = np.asarray(st.growing_plants_mask); ga = np.asarray(st.growing_plants_age)
    d = gp - pos; dist = np.abs(d).sum(1) + (~gm) * 1e6; i = int(np.argmin(dist)); has = float(gm[i])
    return np.array([int(st.timestep) / 10000, has, np.clip(d[i,0]/32, -1, 1) * has, np.clip(d[i,1]/32, -1, 1) * has, min(ga[i], 1000) / 1000 * has], np.float32)

for sd in a.seeds:
    key = jax.random.PRNGKey(sd); obs, st = reset(key); P.reset()
    worst, nbad, n = 0.0, 0, 0
    for t in range(a.max_steps):
        o = np.asarray(obs, np.float32).copy()
        act = P.act(o)  # updates P.s (position, t, plants) for this step
        mine = P.memory_features(None); truth = true_feats(st)
        err = float(np.abs(mine - truth).max()); worst = max(worst, err); nbad += err > 1e-3; n += 1
        obs, st, done = step(key, t, st, act)
        if bool(done): break
    print(f"seed {sd}: {n} steps, feature mismatches {nbad} ({100*nbad/n:.1f}%), worst abs error {worst:.3f}, plants tracked {len(P.s.plants)}")
