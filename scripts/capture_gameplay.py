"""Render real policy trajectories with Craftax's native pixel renderer.

Example: python scripts/capture_gameplay.py --seed 2000
Requires Pillow, available via pip install -e '.[visuals]'.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import jax
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from craftax_rl.env import make_env
from craftax.craftax_classic import constants
from craftax.craftax_classic.constants import BLOCK_PIXEL_SIZE_HUMAN
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", default="policies/run5_curriculum")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--out", default="docs/gameplay")
    args = parser.parse_args()
    # Keep the renderer's generated cache outside the installed package, which
    # may be read-only. This changes storage location, not rendering behavior.
    cache = Path(tempfile.gettempdir()) / "craftax-rl-1.6.1-textures"
    cache.mkdir(parents=True, exist_ok=True)
    constants.TEXTURE_CACHE_FILE = str(cache / "texture_cache_classic.pbz2")
    policy_dir = Path(args.policy_dir).resolve()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("gameplay_policy", policy_dir / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    env, params = make_env()
    reset = jax.jit(lambda key: env.reset(key, params))
    step = jax.jit(lambda key, state, action: env.step_env(key, state, action, params))
    render = jax.jit(make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_HUMAN))
    key = jax.random.PRNGKey(args.seed)
    obs, state = reset(key)
    policy.reset()
    milestones = {
        21: ("crafting.png", "Crafting an iron sword", "after"),
        19: ("diamond.png", "Mining a diamond", "before"),
        11: ("plant.png", "Harvesting a ripe plant", "before"),
    }
    captures = {}
    final_steps = 0
    for t in range(args.max_steps):
        action = int(policy.act(np.asarray(obs, np.float32).copy()))
        assert 0 <= action < 17
        next_obs, next_state, _, done, _ = step(jax.random.fold_in(key, t), state, action)
        old = np.asarray(state.achievements)
        new = np.asarray(next_state.achievements)
        for achievement, (filename, title, timing) in milestones.items():
            if filename in captures or old[achievement] or not new[achievement]:
                continue
            frame_state = state if timing == "before" else next_state
            pixels = np.clip(np.asarray(render(frame_state)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(output / filename)
            captures[filename] = {
                "title": title,
                "frame_step": t if timing == "before" else t + 1,
                "achievement_step": t + 1,
                "timing": f"{timing} the action unlocking the achievement",
                "achievements_in_frame": int(np.asarray(frame_state.achievements).sum()),
                "light_level": float(frame_state.light_level),
                "dimensions": [int(pixels.shape[1]), int(pixels.shape[0])],
            }
            print(f"Captured {filename} at step {captures[filename]['frame_step']}", flush=True)
        obs, state = next_obs, next_state
        final_steps = t + 1
        if bool(done) or len(captures) == len(milestones):
            break
    metadata = {
        "policy_dir": args.policy_dir,
        "seed": args.seed,
        "renderer": "Craftax-Classic native pixel renderer, original lighting and HUD",
        "block_pixel_size": BLOCK_PIXEL_SIZE_HUMAN,
        "steps_simulated": final_steps,
        "captures": captures,
    }
    (output / "captures.json").write_text(json.dumps(metadata, indent=2) + "\n")
    missing = [filename for filename, _, _ in milestones.values() if filename not in captures]
    if missing:
        raise SystemExit(f"Episode did not reach all milestones; missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
