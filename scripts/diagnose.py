"""Why does the policy's episode end? Runs the policy like evaluate.py but watches the env state
and classifies each death. Also reports timing of achievements and what the agent was doing late.

Usage: python scripts/diagnose.py policies/run1_ppo --episodes 100 --seed-start 1000

AI-use note: drafted with Claude Code, reviewed by Yogya Mehrotra.
"""

import argparse
import collections
import multiprocessing as mp
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
LAVA, WATER = 14, 3


def _worker(job):
    policy_dir, seeds, max_steps = job
    import jax
    from craftax_rl.env import PARAMS, make_env

    env, _ = make_env()
    reset = jax.jit(lambda key: env.reset(key, PARAMS))

    @jax.jit
    def step(key, t, state, action):
        obs, state, _, done, _ = env.step_env(jax.random.fold_in(key, t), state, action, PARAMS)
        return obs, state, done

    sys.path.insert(0, os.path.abspath(policy_dir))
    import policy

    out = []
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        obs, state = reset(key)
        policy.reset()
        prev = state
        first_unlock = {}
        hist = []  # (t, health, food, drink, energy, sleeping, light, n_adjacent_mobs)
        for t in range(max_steps):
            a = int(policy.act(np.asarray(obs, np.float32).copy()))
            obs, state, done = step(key, t, state, a)
            ach = np.asarray(state.achievements)
            for i in np.flatnonzero(ach):
                first_unlock.setdefault(int(i), t)
            if t % 50 == 0 or bool(done):
                pp = np.asarray(state.player_position)
                def adj(m):
                    pos = np.asarray(m.position); mask = np.asarray(m.mask)
                    return int(((np.abs(pos - pp).sum(1) <= 1) & mask).sum())
                hist.append((t, int(state.player_health), int(state.player_food),
                             int(state.player_drink), int(state.player_energy),
                             bool(state.is_sleeping), float(state.light_level),
                             adj(state.zombies), adj(state.skeletons),
                             int(state.map[pp[0], pp[1]])))
            if bool(done):
                break
            prev = state
        pp = np.asarray(state.player_position)
        on_lava = int(state.map[pp[0], pp[1]]) == LAVA
        ph = int(prev.player_health)
        h = int(state.player_health)
        food, drink, energy = int(prev.player_food), int(prev.player_drink), int(prev.player_energy)
        zadj = hist[-1][7]; sadj = hist[-1][8]
        if t + 1 >= max_steps and not bool(done):
            cause = "survived_10000"
        elif on_lava:
            cause = "lava"
        elif bool(prev.is_sleeping) and ph - h >= 5:
            cause = "zombie_while_asleep"
        elif ph - h >= 2 and (zadj or sadj):
            cause = "zombie_or_skeleton"
        elif ph - h >= 2:
            cause = "arrow_or_mob_offscreen"
        elif food == 0 or drink == 0 or energy == 0:
            cause = "starved(" + ",".join(n for n, v in (("food", food), ("drink", drink), ("energy", energy)) if v == 0) + ")"
        else:
            cause = "unknown"
        st = getattr(policy, "s", None)
        mode = (st.mode.name + str(getattr(st.mode, "stage", ""))) if st is not None and getattr(st, "mode", None) else "net"
        out.append({"seed": int(seed), "steps": t + 1, "cause": cause, "mode": mode,
                    "ach": int(np.asarray(state.achievements).sum()),
                    "first_unlock": first_unlock, "hist": hist})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_dir")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10000)
    a = ap.parse_args()
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    w = max(1, min(a.workers, len(seeds)))
    with mp.get_context("spawn").Pool(w) as pool:
        res = [r for c in pool.map(_worker, [(a.policy_dir, seeds[i::w], a.max_steps) for i in range(w)]) for r in c]
    causes = collections.Counter(r["cause"] for r in res)
    print("episode end causes:")
    for c, n in causes.most_common():
        steps = [r["steps"] for r in res if r["cause"] == c]
        print(f"  {c:<28} {n:3d}   median steps {int(np.median(steps)):5d}   mean ach {np.mean([r['ach'] for r in res if r['cause']==c]):.1f}")
    modes = collections.Counter(r["mode"] for r in res if r["cause"] != "survived_10000")
    print("\nroutine in control at death:", dict(modes.most_common()))
    print(f"\nsteps: median {int(np.median([r['steps'] for r in res]))}, "
          f"min {min(r['steps'] for r in res)}, max {max(r['steps'] for r in res)}")
    from evaluate import ACH_NAMES
    print("\nmedian step of first unlock (over episodes that unlocked it):")
    for i, name in enumerate(ACH_NAMES):
        ts = [r["first_unlock"][i] for r in res if i in r["first_unlock"]]
        if ts:
            print(f"  {name:<20} n={len(ts):3d}  median t={int(np.median(ts)):5d}")
    # what were stats like in the last sample before death
    last = np.array([r["hist"][-1][1:5] for r in res if r["cause"] != "survived_10000"])
    if len(last):
        print("\nat death: mean health/food/drink/energy =", np.round(last.mean(0), 2).tolist())
        print("fraction with food==0:", round(float((last[:, 1] == 0).mean()), 2),
              " drink==0:", round(float((last[:, 2] == 0).mean()), 2),
              " energy==0:", round(float((last[:, 3] == 0).mean()), 2))
    light = [h[6] for r in res for h in r["hist"][-1:] if r["cause"] != "survived_10000"]
    print("mean light level at death:", round(float(np.mean(light)), 2), "(1=noon, 0=midnight)")
    asleep = np.mean([r["hist"][-1][5] for r in res if r["cause"] != "survived_10000"])
    print("fraction asleep at death:", round(float(asleep), 2))


if __name__ == "__main__":
    main()
