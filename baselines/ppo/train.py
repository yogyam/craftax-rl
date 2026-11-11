"""Compact PPO for the Craftax RL environment. Run: python baselines/ppo/train.py --budget-min 40 --out checkpoints/baseline/weights.npz"""

import argparse
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from craftax_rl.env import NUM_ACTIONS, make_env

GAMMA, LAM, CLIP, VF, ENT, LR = 0.99, 0.95, 0.2, 0.5, 0.01, 3e-4
STEPS, EPOCHS, MINIBATCHES = 64, 4, 8


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256, name="d1")(x))
        x = nn.tanh(nn.Dense(256, name="d2")(x))
        out = nn.Dense(NUM_ACTIONS + 1, name="head")(x)
        return out[..., :NUM_ACTIONS], out[..., NUM_ACTIONS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=50_000_000)
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="checkpoints/baseline/weights.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    env, params = make_env()
    net = Net()
    n = a.num_envs
    # Training keys come from --seed (12345), never PRNGKey(0..9) (the smoke-test seeds).
    rng, k_init, k_reset = jax.random.split(jax.random.PRNGKey(a.seed), 3)
    obs, state = jax.vmap(env.reset, (0, None))(jax.random.split(k_reset, n), params)
    net_params = net.init(k_init, obs[:1])
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(LR, eps=1e-5))
    opt = tx.init(net_params)

    def step_fn(carry, _):
        net_params, obs, state, rng = carry
        rng, ka, ks = jax.random.split(rng, 3)
        logits, value = net.apply(net_params, obs)
        act = jax.random.categorical(ka, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(n), act]
        nobs, nstate, rew, done, _ = jax.vmap(env.step, (0, 0, 0, None))(
            jax.random.split(ks, n), state, act, params
        )
        return (net_params, nobs, nstate, rng), (obs, act, logp, value, rew, done)

    def loss_fn(p, mb):
        o, act, logp_old, adv, ret = mb
        logits, value = net.apply(p, o)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], 1)[:, 0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = jnp.exp(logp - logp_old)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * adv).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return pg + VF * ((value - ret) ** 2).mean() - ENT * ent

    @jax.jit
    def iterate(net_params, opt, obs, state, rng):
        (_, obs, state, rng), (to, ta, tl, tv, tr, td) = jax.lax.scan(
            step_fn, (net_params, obs, state, rng), None, STEPS
        )
        _, last_v = net.apply(net_params, obs)

        def gae(c, x):
            nxt_v, nxt_adv = c
            v, r, d = x
            delta = r + GAMMA * nxt_v * (1 - d) - v
            adv = delta + GAMMA * LAM * (1 - d) * nxt_adv
            return (v, adv), adv

        _, adv = jax.lax.scan(gae, (last_v, jnp.zeros(n)), (tv, tr, td), reverse=True)
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (to, ta, tl, adv, adv + tv))

        def epoch(c, _):
            p, o, rng = c
            rng, k = jax.random.split(rng)
            perm = jax.random.permutation(k, STEPS * n)
            mbs = jax.tree.map(lambda x: x[perm].reshape((MINIBATCHES, -1) + x.shape[1:]), flat)

            def mb_step(c, mb):
                p, o = c
                g = jax.grad(loss_fn)(p, mb)
                up, o = tx.update(g, o)
                return (optax.apply_updates(p, up), o), None

            (p, o), _ = jax.lax.scan(mb_step, (p, o), mbs)
            return (p, o, rng), None

        (net_params, opt, rng), _ = jax.lax.scan(epoch, (net_params, opt, rng), None, EPOCHS)
        return net_params, opt, obs, state, rng, tr.sum() / n, td.sum()

    t0, steps, it = time.time(), 0, 0
    while steps < a.total_steps and time.time() - t0 < a.budget_min * 60:
        net_params, opt, obs, state, rng, rew, eps = iterate(net_params, opt, obs, state, rng)
        steps += STEPS * n
        it += 1
        if it % 10 == 0:
            print(
                f"it {it} steps {steps} t {time.time() - t0:.0f}s "
                f"rew/env {float(rew):.2f} episodes {int(eps)}",
                flush=True,
            )
    p = net_params["params"]
    np.savez(
        a.out,
        w1=np.asarray(p["d1"]["kernel"], np.float32),
        b1=np.asarray(p["d1"]["bias"], np.float32),
        w2=np.asarray(p["d2"]["kernel"], np.float32),
        b2=np.asarray(p["d2"]["bias"], np.float32),
        wp=np.asarray(p["head"]["kernel"], np.float32),
        bp=np.asarray(p["head"]["bias"], np.float32),
    )
    print(f"saved {a.out} after {steps} steps, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
