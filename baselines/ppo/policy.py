"""the Craftax RL environment policy: numpy-only MLP (tanh), weights in weights.npz next to this file.

Weights keys: w1,b1 ... wK,bK (hidden layers) and wp,bp (17 action logits + 1 value). Samples an
action from the softmax with a generator re-seeded in reset(), so every episode is reproducible.

AI-use note: structure follows the reference baseline policy; generalized to K layers with
Claude Code. Reviewed by Yogya Mehrotra.
"""

import os

import numpy as np

_w = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz"))
LAYERS = []
_i = 1
while f"w{_i}" in _w:
    LAYERS.append((_w[f"w{_i}"].astype(np.float32), _w[f"b{_i}"].astype(np.float32)))
    _i += 1
WP, BP = _w["wp"].astype(np.float32), _w["bp"].astype(np.float32)
_rng = np.random.default_rng(0)


def reset():
    global _rng
    _rng = np.random.default_rng(0)


def act(obs):
    h = obs
    for w, b in LAYERS:
        h = np.tanh(h @ w + b)
    logits = (h @ WP + BP)[:17]
    p = np.exp(logits - logits.max())
    return int(_rng.choice(17, p=p / p.sum()))
