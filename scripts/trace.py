"""Step trace of the wrapper policy on one seed: prints mode changes and routine failures.
Usage: python scripts/trace.py policies/run2_shaped --seed 1000 [--verbose]
AI-use note: drafted with Claude Code."""
import argparse, os, sys, collections
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser(); ap.add_argument("policy_dir"); ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--verbose", action="store_true"); ap.add_argument("--max-steps", type=int, default=10000)
a = ap.parse_args()
import jax
from craftax_rl.env import PARAMS, make_env
env, _ = make_env()
reset = jax.jit(lambda key: env.reset(key, PARAMS))
@jax.jit
def step(key, t, state, action):
    obs, state, _, done, _ = env.step_env(jax.random.fold_in(key, t), state, action, PARAMS)
    return obs, state, done
sys.path.insert(0, os.path.abspath(a.policy_dir))
import policy as P
key = jax.random.PRNGKey(a.seed); obs, state = reset(key); P.reset()
names = ["NOOP","L","R","U","D","DO","SLEEP","P_STONE","P_TABLE","P_FURN","P_PLANT","M_WP","M_SP","M_IP","M_WS","M_SS","M_IS"]
last_mode = None; mode_steps = collections.Counter(); start_pos = np.asarray(state.player_position)
for t in range(a.max_steps):
    o = np.asarray(obs, np.float32).copy()
    act = P.act(o)
    m = P.s.mode.name if P.s.mode else "net"
    stage = getattr(P.s.mode, "stage", "") if P.s.mode else ""
    if P.s.mode and getattr(P.s.mode, "walls", None):
        stage = f"{stage}W{len(P.s.mode.walls)}"
    mode_steps[m] += 1
    truth = np.asarray(state.player_position) - start_pos + np.array([P.OFF, P.OFF])
    drift = int(abs(truth - np.array(P.s.pos)).sum())
    c = P.parse(o)
    if m != last_mode or a.verbose or (m != "net" and t % 10 == 0):
        print(f"t={t:5d} {m:8s}{stage!s:3s} act={names[act]:8s} hp={c.health} food={c.food} drink={c.drink} en={c.energy} "
              f"light={c.light:.2f} pos={P.s.pos} drift={drift} inv_stone={c.inv[P.I_STONE]} giveup={P.s.giveup_until}")
        last_mode = m
    obs, state, done = step(key, t, state, act)
    if bool(done):
        print(f"DONE at t={t}, achievements={int(np.asarray(state.achievements).sum())}")
        break
print("mode steps:", dict(mode_steps))
