"""Round-robin ratings for Splendor agents.

    python elo.py --num-players 2 --games 200 --participants random greedy latest
    python elo.py --num-players 3 --games 300 --participants random greedy 'experiments/pool/*.pt'
    python elo.py --participants random greedy --mcts experiments/latest.pt:100

Two players: every pair plays `--games` games, half with the seats swapped.
Three or four: `--games` games spread over random tables of P distinct
participants. Every game contributes one pairwise result per pair of seats
(from `ranks`; equal rank = draw). Ratings are a Bradley-Terry fit (iterative
MM) on those pairwise counts, put on the Elo scale (400 * log10 of the
strength ratio) and anchored so that `random` sits at 0.
"""
import argparse
import glob
import itertools
import json
import os
import time

import numpy as np

from splendor import agents as A


# --- participants -----------------------------------------------------------
def expand(specs):
    """'random' / 'greedy' / 'latest' pass through; paths and globs expand."""
    out, seen = [], set()
    for spec in specs:
        if spec in A.AGENTS:            # 'random'/'greedy' may repeat (more seats)
            out.append(spec)
            continue
        hits = sorted(glob.glob(spec)) if any(c in spec for c in '*?[') else [spec]
        if not hits:
            print(f'note: no file matches {spec!r}, skipping')
        for h in hits:                  # the same checkpoint only plays once
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def label(spec):
    if spec in A.AGENTS or spec == 'latest':
        return spec
    if spec.startswith(A.PUFFER5):   # PufferLib 5.0 weights: 'puffer5:PATH[:H[:L]]'
        from splendor import puffernet
        path, hidden, layers = puffernet.parse_spec(spec)
        base = os.path.basename(path).replace('_weights.bin', '').replace('.bin', '')
        return f'{base}-{hidden}x{layers}'
    return os.path.basename(spec)[:-3] if spec.endswith('.pt') else os.path.basename(spec)


def checkpoint_players(path):
    """Player count a .pt checkpoint was trained for (from its input width),
    or None if it cannot be read. Checkpoints only play at that player count."""
    try:
        import torch
        from splendor import layout as L
        if path.startswith(A.PUFFER5):
            from splendor import puffernet
            arch = puffernet.infer_arch(puffernet.parse_spec(path)[0])
            return arch[0] if arch else None
        if path == 'latest':
            path = max(glob.glob('experiments/**/*.pt', recursive=True),
                       key=os.path.getmtime)
        sd = torch.load(path, map_location='cpu', weights_only=False)
        while isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        w = next(v for k, v in sd.items()
                 if k.endswith('weight') and getattr(v, 'dim', lambda: 0)() == 2)
        return L.num_players_from_obs(int(w.shape[1]))
    except Exception:
        return None


def mcts_agents(entries, num_players, device):
    """`--mcts PATH:SIMS` entries, only if splendor/mcts.py exists (it is
    written by another agent); otherwise a note and no participants."""
    if not entries:
        return []
    try:
        from splendor.mcts import MCTSAgent
    except Exception as e:  # not written yet, or not importable
        print(f'note: --mcts ignored ({type(e).__name__}: {e})')
        return []
    import inspect
    out = []
    for entry in entries:
        path, _, sims = entry.rpartition(':')
        if not path:
            path, sims = entry, '200'
        name = f'mcts-{label(path)}-{sims}'
        kw = dict(num_players=num_players, sims=int(sims), device=device)
        try:
            policy = A.load_policy(path, num_players, device)
            params = inspect.signature(MCTSAgent.__init__).parameters
            agent = MCTSAgent(policy, **{k: v for k, v in kw.items()
                                         if k in params})
            agent.name = name
            out.append((name, agent))
        except Exception as e:
            print(f'note: could not build {name} ({type(e).__name__}: {e})')
    return out


# --- Bradley-Terry ----------------------------------------------------------
def bradley_terry(wins, iters=1000, prior=1.0, tol=1e-12):
    """MM fit of BT strengths from a pairwise win matrix (draws = 0.5 each way).

    `prior` adds a half-win / half-loss against a fixed average opponent so
    that a participant who never won (or never lost) still gets a finite
    rating. Returns strengths normalized to geometric mean 1.
    """
    w = np.asarray(wins, dtype=np.float64)
    np.fill_diagonal(w, 0.0)
    n = w.shape[0]
    games = w + w.T
    p = np.ones(n)
    for _ in range(iters):
        wins_i = w.sum(1) + prior / 2
        denom = (games / (p[:, None] + p[None, :])).sum(1) + prior / (p + 1.0)
        new = np.where(denom > 0, wins_i / np.maximum(denom, 1e-12), p)
        new = np.maximum(new, 1e-12)
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(np.log(new) - np.log(p))) < tol:
            p = new
            break
        p = new
    return p


# --- matches ----------------------------------------------------------------
def pairwise_from_ranks(ranks, seats, wins, draws):
    """Accumulate pairwise wins / draws from a (games, P) rank array.
    `seats[s]` is the participant index sitting in seat s."""
    P = ranks.shape[1]
    for a in range(P):
        for b in range(a + 1, P):
            i, j = seats[a], seats[b]
            ra, rb = ranks[:, a], ranks[:, b]
            wins[i, j] += int((ra < rb).sum())
            wins[j, i] += int((rb < ra).sum())
            d = int((ra == rb).sum())
            draws[i, j] += d
            draws[j, i] += d


def run(participants, num_players=2, games=200, seed=0, max_turns=None,
        tables=None, verbose=True):
    """Play every match-up and return (names, wins, draws) pairwise counts."""
    names = [n for n, _ in participants]
    k = len(names)
    if k < num_players:
        raise SystemExit(f'need at least {num_players} participants, got {k}')
    wins = np.zeros((k, k), dtype=np.int64)
    draws = np.zeros((k, k), dtype=np.int64)
    rng = np.random.default_rng(seed)
    agents = [a for _, a in participants]

    def play(seats, n, s):
        res = A.play_games([agents[i] for i in seats], n, num_players,
                           seed=int(s), max_turns=max_turns)
        ok = res['finished']
        pairwise_from_ranks(res['ranks'][ok], seats, wins, draws)
        return int(ok.sum())

    t0 = time.time()
    if num_players == 2:
        for i, j in itertools.combinations(range(k), 2):
            half = games // 2
            if half:
                play((i, j), half, rng.integers(1 << 30))
            if games - half:
                play((j, i), games - half, rng.integers(1 << 30))
            if verbose:
                print(f'  {names[i]} vs {names[j]}: '
                      f'{wins[i, j]}-{wins[j, i]}-{draws[i, j]} (W-L-D)')
    else:
        n_tables = tables or max(1, min(games // 8 or 1, 64))
        per = max(1, games // n_tables)
        for t in range(n_tables):
            seats = tuple(rng.choice(k, num_players, replace=False))
            n = play(seats, per, rng.integers(1 << 30))
            if verbose:
                print(f'  table {t + 1}/{n_tables}: '
                      f'{" ".join(names[i] for i in seats)} ({n} games)')
    if verbose:
        print(f'  played in {time.time() - t0:.1f}s')
    return names, wins, draws


def ratings(names, wins, draws, anchor='random', prior=1.0):
    """Bradley-Terry -> Elo, anchored at `anchor` (or mean 0 if absent)."""
    score = wins + 0.5 * draws
    p = bradley_terry(score, prior=prior)
    elo = 400.0 * np.log10(p)
    if anchor in names:
        elo = elo - elo[names.index(anchor)]
    else:
        elo = elo - elo.mean()
    played = (wins + wins.T + draws).sum(1)
    out = {}
    for i, n in enumerate(names):
        out[n] = dict(elo=round(float(elo[i]), 1), games=int(played[i]),
                      wins=int(wins[i].sum()), draws=int(draws[i].sum()))
    return out


def print_table(table):
    order = sorted(table, key=lambda n: -table[n]['elo'])
    print(f'\n{"name":<28}{"elo":>9}{"games":>8}{"wins":>7}{"draws":>7}{"win%":>8}')
    print('-' * 67)
    for n in order:
        r = table[n]
        pct = 100.0 * r['wins'] / max(r['games'], 1)
        print(f'{n:<28}{r["elo"]:>9.1f}{r["games"]:>8}{r["wins"]:>7}'
              f'{r["draws"]:>7}{pct:>7.1f}%')
    print()


def main():
    ap = argparse.ArgumentParser(description='Elo ratings for Splendor agents')
    ap.add_argument('--participants', nargs='+',
                    default=['random', 'greedy', 'latest'],
                    help="'random', 'greedy', 'latest', .pt paths or globs")
    ap.add_argument('--mcts', action='append', default=[],
                    metavar='PATH:SIMS', help='add an MCTS agent (repeatable)')
    ap.add_argument('--num-players', type=int, default=2)
    ap.add_argument('--games', type=int, default=200,
                    help='games per pair (2 players) or in total (3-4)')
    ap.add_argument('--tables', type=int, default=None,
                    help='number of random tables for 3-4 players')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-turns', type=int, default=None)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prior', type=float, default=1.0,
                    help='virtual games regularizing the Bradley-Terry fit')
    ap.add_argument('--out', type=str, default='elo.json')
    args = ap.parse_args()

    specs = expand(args.participants)
    participants = []
    for i, spec in enumerate(specs):
        try:
            agent = A.make_agent(spec, args.num_players, args.device,
                                 seed=args.seed + i,
                                 temperature=args.temperature,
                                 name=label(spec))
        except Exception as e:
            trained = checkpoint_players(spec)
            hint = (f' - trained for {trained} players'
                    if trained and trained != args.num_players else '')
            print(f'note: skipping {spec}{hint} ({type(e).__name__}: '
                  f'{str(e).splitlines()[0]})')
            continue
        taken = [n for n, _ in participants]
        if agent.name in taken:         # e.g. 'random random' for a 3rd seat
            agent.name += f'#{sum(n.split("#")[0] == agent.name for n in taken) + 1}'
        participants.append((agent.name, agent))
    participants += mcts_agents(args.mcts, args.num_players, args.device)
    if len(participants) < args.num_players:
        raise SystemExit('not enough participants to play a game')

    print(f'{len(participants)} participants, {args.num_players} players: '
          + ', '.join(n for n, _ in participants))
    names, wins, draws = run(participants, args.num_players, args.games,
                             seed=args.seed, max_turns=args.max_turns,
                             tables=args.tables)
    table = ratings(names, wins, draws)
    print_table(table)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(table, f, indent=2)
        print(f'wrote {args.out}')
    return table


if __name__ == '__main__':
    main()
