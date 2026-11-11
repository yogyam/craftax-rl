"""Evaluate policy folders on seeded worlds without Docker.

Each episode uses fold_in(world_key, step) for transition randomness. Reports mean
achievement count, episode-bootstrap confidence intervals, unlock rates and latency.

Example: python scripts/evaluate.py policies/run5_curriculum --episodes 100 --workers 4

AI-use note: drafted with Claude Code, reviewed by Yogya Mehrotra;
standalone project packaging updated with Codex.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")

ACH_NAMES = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie", "collect_stone",
    "place_stone", "eat_plant", "defeat_skeleton", "make_stone_pickaxe", "make_stone_sword",
    "wake_up", "place_furnace", "collect_coal", "collect_iron", "collect_diamond",
    "make_iron_pickaxe", "make_iron_sword",
]


def _worker(job):
    policy_dir, seeds, max_steps = job
    import jax

    sys.path.insert(0, SRC_DIR)
    from craftax_rl.env import PARAMS, make_env

    env, _ = make_env()
    reset = jax.jit(lambda key: env.reset(key, PARAMS))

    @jax.jit
    def step(key, t, state, action):
        obs, state, _, done, _ = env.step_env(jax.random.fold_in(key, t), state, action, PARAMS)
        return obs, state, done

    sys.path.insert(0, os.path.abspath(policy_dir))
    import policy

    reset_fn = getattr(policy, "reset", None)
    out = []
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        obs, state = reset(key)
        if reset_fn:
            reset_fn()
        steps, pol_t = 0, 0.0
        for t in range(max_steps):
            o = np.asarray(obs, np.float32).copy()
            t0 = time.perf_counter()
            a = int(policy.act(o))
            pol_t += time.perf_counter() - t0
            if not 0 <= a < 17:
                raise ValueError(f"act() returned {a}")
            obs, state, done = step(key, t, state, a)
            steps = t + 1
            if bool(done):
                break
        ach = np.asarray(state.achievements).astype(int).tolist()
        out.append({"seed": int(seed), "achievements": ach, "steps": steps,
                    "ms_per_act": 1000.0 * pol_t / steps})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy_dir")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="use debugging seeds 0..9")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=10000)
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.episodes < 1 or a.max_steps < 1 or a.workers < 1:
        ap.error("episodes, max-steps and workers must be positive")

    seeds = list(range(10)) if a.smoke else list(range(a.seed_start, a.seed_start + a.episodes))
    w = max(1, min(a.workers, len(seeds)))
    jobs = [(a.policy_dir, seeds[i::w], a.max_steps) for i in range(w)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(w) as pool:
        results = [r for chunk in pool.map(_worker, jobs) for r in chunk]
    results.sort(key=lambda r: seeds.index(r["seed"]))
    elapsed = time.time() - t0

    sys.path.insert(0, SRC_DIR)
    from craftax_rl.scoring import summarize

    counts = [sum(r["achievements"]) for r in results]
    mean, lo, hi = summarize(counts)
    rates = np.mean([r["achievements"] for r in results], axis=0)
    if not a.quiet:
        for r in results:
            print(f"seed {r['seed']:>6}: {sum(r['achievements']):2d} ach, {r['steps']:5d} steps")
    print("\nper-achievement unlock rate:")
    for i in np.argsort(-rates):
        print(f"  {ACH_NAMES[i]:<20} {100 * rates[i]:5.1f}%")
    print(f"\nmean {mean:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  over {len(results)} episodes")
    print(f"avg steps {np.mean([r['steps'] for r in results]):.0f}, "
          f"policy {np.mean([r['ms_per_act'] for r in results]):.3f} ms/act, "
          f"wall {elapsed:.0f}s")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"policy_dir": a.policy_dir, "seeds": seeds, "mean": mean, "ci": [lo, hi],
                       "rates": dict(zip(ACH_NAMES, rates.tolist())), "episodes": results}, f,
                      indent=1)


if __name__ == "__main__":
    main()
