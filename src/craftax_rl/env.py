"""Craftax-Classic parameter variant used throughout the five experiments.

The Craftax dependency is pinned; the game implementation is not patched.
"""

from craftax.craftax_classic.envs.craftax_state import EnvParams
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

NUM_ACTIONS = 17

ENV_VERSION = "v1"
PARAMS = EnvParams(
    day_length=240,
    cow_health=4,
    zombie_health=7,
    skeleton_health=4,
    spawn_zombie_night_chance=0.14,
    spawn_skeleton_chance=0.07,
)


def make_env() -> tuple[CraftaxClassicSymbolicEnv, EnvParams]:
    return CraftaxClassicSymbolicEnv(), PARAMS
