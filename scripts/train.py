"""Vectorized JAX/Flax PPO with shaping, memory features and late-state resets.

Adapted from the reference PPO loop. Run recipes are in docs/training.md.
Checkpoints contain network weights; optimizer and environment state are not saved.

AI-use note: reference loop by the starter-kit authors; trainer extensions drafted
with Claude Code, reviewed by Yogya Mehrotra; packaging updated with Codex.
"""

import argparse
import os
import sys
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from craftax_rl.env import NUM_ACTIONS, make_env  # noqa: E402


class Net(nn.Module):
    width: int
    layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.layers):
            x = nn.tanh(nn.Dense(self.width, name=f"d{i + 1}")(x))
        out = nn.Dense(NUM_ACTIONS + 1, name="head")(x)
        return out[..., :NUM_ACTIONS], out[..., NUM_ACTIONS]


def save_weights(path, net_params, layers):
    p = net_params["params"]
    d = {}
    for i in range(1, layers + 1):
        d[f"w{i}"] = np.asarray(p[f"d{i}"]["kernel"], np.float32)
        d[f"b{i}"] = np.asarray(p[f"d{i}"]["bias"], np.float32)
    d["wp"] = np.asarray(p["head"]["kernel"], np.float32)
    d["bp"] = np.asarray(p["head"]["bias"], np.float32)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=10_000_000_000)
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="checkpoints/weights.npz")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--anneal-lr", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--ckpt-min", type=float, default=20.0)
    ap.add_argument("--eval-every-min", type=float, default=30.0)
    ap.add_argument("--eval-envs", type=int, default=128)
    ap.add_argument("--eval-steps", type=int, default=10000)
    ap.add_argument("--init-from", default=None, help="npz to warm-start from (same arch)")
    ap.add_argument("--keep-ckpts", action="store_true", help="also keep a numbered copy of every checkpoint")
    ap.add_argument("--late-resets", type=float, default=0.0,
                    help="probability that an episode end restarts from a saved late-game state "
                         "(snapshot taken when an env first owns both iron tools) instead of a fresh world")
    ap.add_argument("--late-buffer", type=int, default=256, help="number of saved late-game states")
    ap.add_argument("--plant-feats", action="store_true",
                    help="append 5 memory features to obs: t/10000, has_plant, d_row/32, d_col/32, age/1000 "
                         "(nearest growing plant). policy.py must compute the same 5 from its own tracking")
    ap.add_argument("--shape-needs", type=float, default=0.0,
                    help="extra reward per step in which drink, food or energy went UP (teaches upkeep)")
    ap.add_argument("--shape-deficit", type=int, default=99,
                    help="only pay the upkeep reward when the stat was at or below this before going up "
                         "(default 99 = always); e.g. 4 rewards fixing a deficit, not farming")
    ap.add_argument("--deficit-drink", type=int, default=None, help="per-stat override of --shape-deficit")
    ap.add_argument("--deficit-food", type=int, default=None)
    ap.add_argument("--deficit-energy", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    STEPS, EPOCHS, MINIBATCHES = a.steps, a.epochs, a.minibatches

    env, params = make_env()
    net = Net(a.width, a.layers)
    n = a.num_envs

    def plant_feats(obs, state):
        """Reconstructed by the inference tracker: see policies/run4_memory/policy.py."""
        d = state.growing_plants_positions - state.player_position
        dist = jnp.abs(d).sum(1) + (~state.growing_plants_mask) * 1e6
        i = jnp.argmin(dist)
        has = state.growing_plants_mask[i].astype(jnp.float32)
        dr = jnp.clip(d[i, 0] / 32.0, -1, 1) * has
        dc = jnp.clip(d[i, 1] / 32.0, -1, 1) * has
        age = jnp.minimum(state.growing_plants_age[i], 1000) / 1000.0 * has
        t = state.timestep / 10000.0
        return jnp.concatenate([obs, jnp.stack([t, has, dr, dc, age]).astype(obs.dtype)])

    aug = jax.vmap(plant_feats) if a.plant_feats else (lambda o, st: o)
    # Training keys come from --seed (12345), never PRNGKey(0..9) (the smoke-test seeds).
    rng, k_init, k_reset = jax.random.split(jax.random.PRNGKey(a.seed), 3)
    obs, state = jax.vmap(env.reset, (0, None))(jax.random.split(k_reset, n), params)
    obs = aug(obs, state)
    K = a.late_buffer
    # late-game bank: K env states (initially copies of fresh worlds, marked invalid) + a write pointer
    buf = jax.tree.map(lambda x: jnp.repeat(x[:1], K, axis=0), state)
    buf_valid = jnp.zeros(K, bool)
    buf_ptr = jnp.int32(0)
    net_params = net.init(k_init, obs[:1])
    if a.init_from:
        w = np.load(a.init_from)
        w1 = jnp.asarray(w["w1"])
        if w1.shape[0] < obs.shape[1]:  # warm start from a network without the extra inputs
            w1 = jnp.concatenate([w1, jnp.zeros((obs.shape[1] - w1.shape[0], w1.shape[1]), w1.dtype)])
            print(f"padded w1 {w['w1'].shape} -> {w1.shape} with zero rows for the new features")
        p = {f"d{i}": {"kernel": (w1 if i == 1 else jnp.asarray(w[f"w{i}"])), "bias": jnp.asarray(w[f"b{i}"])}
             for i in range(1, a.layers + 1)}
        p["head"] = {"kernel": jnp.asarray(w["wp"]), "bias": jnp.asarray(w["bp"])}
        net_params = {"params": p}
        print(f"warm-started from {a.init_from}")
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.inject_hyperparams(optax.adam)(learning_rate=a.lr, eps=1e-5),
    )
    opt = tx.init(net_params)

    def bsel(mask, x, y):
        """Per-env select over a pytree leaf (mask has shape (n,))."""
        return jnp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)

    def step_fn(carry, _):
        net_params, obs, state, rng, buf, buf_valid, buf_ptr = carry
        rng, ka, ks, kr1, kr2 = jax.random.split(rng, 5)
        logits, value = net.apply(net_params, obs)
        act = jax.random.categorical(ka, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(n), act]
        nobs, nstate, rew, done, info = jax.vmap(env.step, (0, 0, 0, None))(
            jax.random.split(ks, n), state, act, params
        )
        if a.late_resets > 0:
            # 1) snapshot: an env that just got its iron sword (and has the pickaxe) goes into the bank
            newly = ((nstate.inventory.iron_sword >= 1) & (nstate.inventory.iron_pickaxe >= 1)
                     & (state.inventory.iron_sword < 1) & ~done)
            any_new = newly.any()
            j = jnp.argmax(newly)
            slot = buf_ptr % K
            snap = jax.tree.map(lambda x: x[j], nstate)
            buf = jax.tree.map(lambda b, v: jnp.where(any_new, b.at[slot].set(v), b), buf, snap)
            buf_valid = buf_valid.at[slot].set(buf_valid[slot] | any_new)
            buf_ptr = buf_ptr + any_new.astype(jnp.int32)
            # 2) restore: some finished episodes restart from a random saved late-game state
            idx = jax.random.randint(kr1, (n,), 0, K)
            use = done & (jax.random.uniform(kr2, (n,)) < a.late_resets) & buf_valid[idx]
            cand = jax.tree.map(lambda b: b[idx], buf)
            nstate = jax.tree.map(lambda c_, s_: bsel(use, c_, s_), cand, nstate)
            nobs = bsel(use, jax.vmap(env.get_obs)(cand), nobs)
        nobs = aug(nobs, nstate)
        if a.shape_needs:
            ld = a.shape_deficit if a.deficit_drink is None else a.deficit_drink
            lf = a.shape_deficit if a.deficit_food is None else a.deficit_food
            le = a.shape_deficit if a.deficit_energy is None else a.deficit_energy
            up = (
                ((nstate.player_drink > state.player_drink) & (state.player_drink <= ld))
                | ((nstate.player_food > state.player_food) & (state.player_food <= lf))
                | ((nstate.player_energy > state.player_energy) & (state.player_energy <= le))
            )
            rew = rew + a.shape_needs * up * (1 - done)
        # info["Achievements/*"] is 100*flag on done steps, 0 otherwise -> achievements per episode
        ach = sum(v for k, v in info.items() if k.startswith("Achievements/")) / 100.0
        return (net_params, nobs, nstate, rng, buf, buf_valid, buf_ptr), (obs, act, logp, value, rew, done, ach)

    def loss_fn(p, mb):
        o, act, logp_old, adv, ret = mb
        logits, value = net.apply(p, o)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], 1)[:, 0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ratio = jnp.exp(logp - logp_old)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - a.clip, 1 + a.clip) * adv).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        return pg + a.vf * ((value - ret) ** 2).mean() - a.ent * ent

    @jax.jit
    def iterate(net_params, opt, obs, state, rng, lr, buf, buf_valid, buf_ptr):
        (_, obs, state, rng, buf, buf_valid, buf_ptr), (to, ta, tl, tv, tr, td, tach) = jax.lax.scan(
            step_fn, (net_params, obs, state, rng, buf, buf_valid, buf_ptr), None, STEPS
        )
        _, last_v = net.apply(net_params, obs)

        def gae(c, x):
            nxt_v, nxt_adv = c
            v, r, d = x
            delta = r + a.gamma * nxt_v * (1 - d) - v
            adv = delta + a.gamma * a.lam * (1 - d) * nxt_adv
            return (v, adv), adv

        _, adv = jax.lax.scan(gae, (last_v, jnp.zeros(n)), (tv, tr, td), reverse=True)
        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (to, ta, tl, adv, adv + tv))
        # set the (possibly annealed) learning rate on the inject_hyperparams(adam) state
        opt = (opt[0], opt[1]._replace(hyperparams={**opt[1].hyperparams, "learning_rate": lr}))

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
        n_done = td.sum()
        return (net_params, opt, obs, state, rng, tr.sum() / n, n_done, tach.sum() / jnp.maximum(n_done, 1),
                buf, buf_valid, buf_ptr)

    E, ES = a.eval_envs, a.eval_steps

    @jax.jit
    def evaluate(net_params, key):
        keys = jax.random.split(key, E)
        obs, state = jax.vmap(env.reset, (0, None))(keys, params)
        obs = aug(obs, state)

        def body(c, _):
            obs, state, finished, rng = c
            rng, ka, ks = jax.random.split(rng, 3)
            logits, _ = net.apply(net_params, obs)
            act = jax.random.categorical(ka, logits)
            nobs, nstate, _, done, _ = jax.vmap(env.step_env, (0, 0, 0, None))(
                jax.random.split(ks, E), state, act, params
            )
            nobs = aug(nobs, nstate)
            keep = lambda new, old: jnp.where(  # noqa: E731
                finished.reshape((-1,) + (1,) * (new.ndim - 1)), old, new
            )
            state = jax.tree.map(keep, nstate, state)
            obs = keep(nobs, obs)
            return (obs, state, finished | done, rng), None

        (obs, state, finished, _), _ = jax.lax.scan(
            body, (obs, state, jnp.zeros(E, bool), key), None, ES
        )
        return state.achievements.sum(-1).mean(), state.achievements.mean(0)

    eval_key = jax.random.PRNGKey(10_000)  # fixed validation worlds, never the smoke-test seeds 0..9
    log_path = a.out + ".log.csv"
    with open(log_path, "a") as f:
        f.write("it,steps,t_s,lr,rew_per_env,episodes,train_ach,eval_ach\n")

    t0, steps, it = time.time(), 0, 0
    last_ckpt, last_eval = t0, t0
    budget_s = a.budget_min * 60
    print(f"arch {a.layers}x{a.width}, envs {n}, budget {a.budget_min} min, anneal {a.anneal_lr}")
    while steps < a.total_steps and time.time() - t0 < budget_s:
        frac = min(1.0, (time.time() - t0) / budget_s)
        lr = a.lr * (1.0 - frac) if a.anneal_lr else a.lr
        net_params, opt, obs, state, rng, rew, eps, tach, buf, buf_valid, buf_ptr = iterate(
            net_params, opt, obs, state, rng, jnp.float32(lr), buf, buf_valid, buf_ptr
        )
        rew, eps, tach = float(rew), int(eps), float(tach)  # blocks: keeps timings honest
        steps += STEPS * n
        it += 1
        now = time.time()
        eval_ach = ""
        if now - last_eval > a.eval_every_min * 60:
            m, per = evaluate(net_params, eval_key)
            eval_ach = f"{float(m):.2f}"
            top = np.asarray(per)
            print(f"  [eval] mean achievements {float(m):.2f} on {E} validation worlds; "
                  f"rates: " + " ".join(f"{x:.2f}" for x in top), flush=True)
            last_eval = time.time()
        if now - last_ckpt > a.ckpt_min * 60:
            save_weights(a.out, net_params, a.layers)
            if a.keep_ckpts:  # numbered copy, so the best checkpoint can be picked afterwards
                save_weights(a.out.replace(".npz", f"_{int((now - t0) // 60):04d}min.npz"), net_params, a.layers)
            last_ckpt = time.time()
        if it % 10 == 0 or eval_ach:
            msg = (f"it {it} steps {steps / 1e6:.1f}M t {now - t0:.0f}s lr {lr:.2e} "
                   f"rew/env {float(rew):.2f} episodes {int(eps)} train_ach {float(tach):.2f}"
                   + (f" bank {int(buf_valid.sum())}/{K}" if a.late_resets > 0 else ""))
            print(msg, flush=True)
            with open(log_path, "a") as f:
                f.write(f"{it},{steps},{now - t0:.0f},{lr:.3e},{float(rew):.3f},{int(eps)},"
                        f"{float(tach):.3f},{eval_ach}\n")
    save_weights(a.out, net_params, a.layers)
    m, per = evaluate(net_params, eval_key)
    print(f"saved {a.out} after {steps} steps, {time.time() - t0:.0f}s; final eval {float(m):.2f}")


if __name__ == "__main__":
    main()
