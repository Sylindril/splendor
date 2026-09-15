"""AlphaZero-style self-play training for Splendor.

Every seat of every game is played by the same batched PUCT search
(`splendor.mcts.MCTS`) over the current network.  Each decision - including the
discard / noble sub-phase steps, which are ordinary decision nodes - records the
mover's observation row, the normalized root visit distribution, and, once the
game finishes, that seat's outcome z (+1 win / -1 loss / 0 draw) taken from the
terminal rewards vector.  The network is then trained with policy cross-entropy
against the visit distribution plus value MSE against z, and `iter_{k}.pt` is
written (a plain state dict, loadable by `splendor.agents.load_policy`).

    python alphazero.py --games 64 --sims 100 --iters 2 --epochs 2
    python alphazero.py --num-players 3 --init experiments/latest.pt --device mps
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from splendor import layout as L
from splendor import policy as Pol
from splendor import agents as A
from splendor.mcts import MCTS
from splendor.splendor import Splendor


def matching_checkpoints(num_players, root='experiments'):
    """Every .pt under `root` playable at `num_players`, newest first.

    `agents.load_policy('latest')` takes the newest checkpoint of *any* player
    count, which breaks as soon as runs for several table sizes share
    experiments/; this filters by the observation width first.
    """
    want = L.obs_size(num_players)
    out = []
    for f in sorted(glob.glob(os.path.join(root, '**', '*.pt'), recursive=True),
                    key=os.path.getmtime, reverse=True):
        try:
            sd = torch.load(f, map_location='cpu', weights_only=False)
            while isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
            w = sd.get('encoder.0.weight')
            if w is not None and w.dim() == 2 and int(w.shape[1]) == want:
                out.append(f)
        except Exception:
            continue
    return out


def latest_checkpoint(num_players, root='experiments'):
    """Newest checkpoint playable at `num_players`, or None."""
    found = matching_checkpoints(num_players, root)
    return found[0] if found else None


def build_net(init, num_players, device):
    """`--init` is 'latest', a checkpoint path, or 'scratch'/'none'/''."""
    if init == 'latest':
        init = latest_checkpoint(num_players)
        if init is None:
            print(f'no {num_players}-player checkpoint under experiments/')
    if init and init not in ('scratch', 'none', 'random'):
        net = A.load_policy(init, num_players, device)
        print(f'warm start from {init}')
    else:
        net = Pol.Big(A._Spaces(num_players)).to(device)
        print('fresh Big policy')
    return net


def self_play(net, num_players=2, games=64, sims=100, device='cpu', seed=0,
              determinize=True, temp_turns=15, max_turns=None, verbose=True):
    """Play `games` games to the end with MCTS on every seat.

    Returns (obs (n, OBS_N) uint8, pi (n, 72) float32, z (n,) float32, stats).
    """
    P = num_players
    env = Splendor(num_envs=games, num_players=P, seed=seed,
                   max_turns=max_turns, report_interval=10**9)
    env.reset(seed=seed)
    mcts = MCTS(net, P, sims=sims, determinize=determinize, noise=True,
                device=device, seed=seed, capacity=games, max_turns=max_turns)
    rng = np.random.default_rng(seed)

    actions = np.full(env.num_agents, L.NOOP, dtype=np.int32)
    done = np.zeros(games, dtype=bool)
    pending = [[] for _ in range(games)]
    obs_out, pi_out, z_out = [], [], []
    tm_off, turn_off = L.to_move_offset(P), L.turn_offset(P)
    n_sims = 0
    start = time.perf_counter()

    while not done.all():
        active = np.flatnonzero(~done)
        tomove = env.observations[:, tm_off].reshape(games, P)
        seats = tomove[active].argmax(axis=1)
        states = [env.get_state(int(g)) for g in active]
        visits, _ = mcts.search(states, seats)
        n_sims += len(active) * sims
        turns = env.observations[active * P, turn_off]
        actions[:] = L.NOOP
        for i, g in enumerate(active):
            g, s = int(g), int(seats[i])
            row = g * P + s
            v = visits[i].astype(np.float64)
            total = v.sum()
            if total <= 0:                      # degenerate (sims == 0)
                legal = np.flatnonzero(env.observations[row, L.mask_offset(P):turn_off])
                actions[row] = legal[0] if len(legal) else L.PASS
                continue
            pi = (v / total).astype(np.float32)
            pending[g].append((env.observations[row].copy(), pi, s))
            if turns[i] < temp_turns:
                actions[row] = rng.choice(L.NUM_ACTIONS, p=v / total)
            else:
                actions[row] = int(np.argmax(v))
        env.step(actions)
        just = env.terminals.reshape(games, P)[:, 0].astype(bool) & ~done
        for g in np.flatnonzero(just):
            r = env.rewards.reshape(games, P)[g]
            z = np.where(r > 0.5, 1.0, np.where(r < -0.5, -1.0, 0.0)).astype(np.float32)
            for o, pi, s in pending[int(g)]:
                obs_out.append(o)
                pi_out.append(pi)
                z_out.append(z[s])
            pending[int(g)] = []
        done |= just
        if verbose and just.any() and not done.all():
            print(f'  self-play: {int(done.sum())}/{games} games, '
                  f'{len(obs_out)} samples', end='\r', flush=True)

    elapsed = time.perf_counter() - start
    mcts.close()
    env.close()
    zs = np.asarray(z_out, dtype=np.float32)
    stats = dict(samples=len(zs), seconds=elapsed, sims=n_sims,
                 sims_per_s=n_sims / max(elapsed, 1e-9),
                 win_frac=float((zs > 0).mean()) if len(zs) else 0.0,
                 draw_frac=float((zs == 0).mean()) if len(zs) else 0.0)
    if verbose:
        print(f'  self-play: {games} games, {len(zs)} samples, '
              f'{elapsed:.1f}s, {stats["sims_per_s"]:,.0f} sims/s')
    return (np.stack(obs_out) if obs_out else np.zeros((0, env.obs_n), np.uint8),
            np.stack(pi_out) if pi_out else np.zeros((0, L.NUM_ACTIONS), np.float32),
            zs, stats)


def train(net, obs, pi, z, epochs=2, lr=1e-3, batch_size=256, device='cpu',
          value_coef=1.0, verbose=True):
    """Policy cross-entropy against the visit distribution + value MSE."""
    if len(z) == 0:
        return {}
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    obs_t = torch.from_numpy(obs)
    pi_t = torch.from_numpy(pi)
    z_t = torch.from_numpy(z)
    n = len(z)
    net.train()
    stats = {}
    for e in range(epochs):
        perm = torch.randperm(n)
        pl = vl = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ob = obs_t[idx].to(device)
            tp = pi_t[idx].to(device)
            tz = z_t[idx].to(device)
            logits, value = net.forward_eval(ob, {})
            # illegal logits are -1e8 (finite), and the target is 0 there
            logp = torch.log_softmax(logits.float(), dim=1)
            p_loss = -(tp * logp).sum(dim=1).mean()
            v_loss = torch.nn.functional.mse_loss(value.float().reshape(-1), tz)
            loss = p_loss + value_coef * v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            pl += float(p_loss.detach())
            vl += float(v_loss.detach())
            nb += 1
        stats = dict(policy_loss=pl / max(nb, 1), value_loss=vl / max(nb, 1))
        if verbose:
            print(f'  epoch {e + 1}/{epochs}: policy {stats["policy_loss"]:.4f} '
                  f'value {stats["value_loss"]:.4f}')
    net.eval()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--num-players', type=int, default=2)
    ap.add_argument('--games', type=int, default=64, help='self-play games per iteration')
    ap.add_argument('--sims', type=int, default=100, help='MCTS simulations per move')
    ap.add_argument('--iters', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--init', default='latest',
                    help="'latest', a checkpoint path, or 'scratch'")
    ap.add_argument('--out', default='experiments/az')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--temp-turns', type=int, default=15)
    ap.add_argument('--max-turns', type=int, default=None)
    ap.add_argument('--no-determinize', action='store_true')
    ap.add_argument('--replay-iters', type=int, default=2,
                    help='how many recent iterations of samples to train on')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    P = args.num_players
    os.makedirs(args.out, exist_ok=True)
    net = build_net(args.init, P, args.device)
    replay = []

    for k in range(args.iters):
        print(f'=== iteration {k} ===')
        obs, pi, z, stats = self_play(
            net, num_players=P, games=args.games, sims=args.sims,
            device=args.device, seed=args.seed + 1000 * k,
            determinize=not args.no_determinize, temp_turns=args.temp_turns,
            max_turns=args.max_turns)
        replay.append((obs, pi, z))
        replay = replay[-max(1, args.replay_iters):]
        ob = np.concatenate([r[0] for r in replay])
        pp = np.concatenate([r[1] for r in replay])
        zz = np.concatenate([r[2] for r in replay])
        print(f'  training on {len(zz)} samples '
              f'(decisive {1 - stats["draw_frac"]:.2f})')
        train(net, ob, pp, zz, epochs=args.epochs, lr=args.lr,
              batch_size=args.batch_size, device=args.device)
        path = os.path.join(args.out, f'iter_{k}.pt')
        tmp = path + '.tmp'
        torch.save({k2: v.cpu() for k2, v in net.state_dict().items()}, tmp)
        os.replace(tmp, path)
        print(f'  saved {path}')
    return net


if __name__ == '__main__':
    main()
