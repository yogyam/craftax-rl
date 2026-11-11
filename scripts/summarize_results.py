"""Validate archived results and regenerate the main experiment summary."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from craftax_rl.scoring import summarize


def read_and_validate(path):
    data = json.loads(path.read_text())
    episodes = data["episodes"]
    seeds = [e["seed"] for e in episodes]
    assert seeds == data["seeds"] and len(set(seeds)) == len(seeds), path
    ach = np.array([e["achievements"] for e in episodes])
    assert ach.shape == (len(episodes), 22) and np.isin(ach, [0, 1]).all(), path
    assert all(0 < e["steps"] <= 10000 for e in episodes), path
    assert np.isclose(ach.sum(axis=1).mean(), data["mean"]), path
    assert np.allclose(ach.mean(axis=0), list(data["rates"].values())), path
    assert np.allclose(summarize(ach.sum(axis=1))[1:], data["ci"]), path
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="optional JSON summary path")
    args = parser.parse_args()
    cache = {}
    for folder in ("evaluations", "ablations"):
        for path in sorted((ROOT / "results" / folder).glob("*.json")):
            cache[path.stem] = read_and_validate(path)
    manifest = json.loads((ROOT / "results/checkpoints.json").read_text())
    for folder, metadata in manifest.items():
        path = ROOT / folder / "weights.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["weights_sha256"], path
    print(f"Validated {len(cache)} evaluations and {len(manifest)} checkpoint hashes.\n")
    summary = []
    print("| Policy | Episodes | Mean | 95% bootstrap CI | Plant | Diamond | Avg steps |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for label, files in [
        ("Baseline", ["baseline_1000"]),
        ("Run 1", ["run1_2000"]),
        ("Run 2", ["run2_2000"]),
        ("Run 3", ["run3_2000", "run3_3000"]),
        ("Run 4", ["run4_2000", "run4_3000"]),
        ("Run 5", ["run5_2000", "run5_3000"]),
    ]:
        episodes = [e for file in files for e in cache[file]]
        scores = np.array([sum(e["achievements"]) for e in episodes])
        mean, lo, hi = summarize(scores)
        rates = np.array([e["achievements"] for e in episodes]).mean(axis=0)
        avg_steps = np.mean([e["steps"] for e in episodes])
        row = dict(policy=label, episodes=len(episodes), mean=mean, ci=[lo, hi],
                   plant=float(rates[11]), diamond=float(rates[19]), avg_steps=float(avg_steps),
                   seeds=[e["seed"] for e in episodes])
        summary.append(row)
        print(f"| {label} | {len(episodes)} | {mean:.3f} | [{lo:.2f}, {hi:.2f}] | "
              f"{rates[11]:.1%} | {rates[19]:.1%} | {avg_steps:.0f} |")
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
