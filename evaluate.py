"""Evaluate a Splendor policy against random / greedy / another checkpoint.

    python evaluate.py --load-model-path latest --games 1000
    python evaluate.py --load-model-path latest --opponent greedy --num-players 3
    python evaluate.py --load-model-path latest --render          # one ANSI game

Seat 0 is the policy under test, the other seats are the opponent. A thin
front-end over `splendor.agents.play_games`, which plays every game in
parallel until each has finished once (the start seat is random, so seat
order carries no advantage).
"""
import argparse
import sys

import numpy as np

from splendor import agents as A
from splendor import layout as L
from splendor.splendor import Splendor


def render_game(agents, num_players=2, seed=0, max_turns=None):
    """Play a single game in one env, printing the ANSI board every turn."""
    env = Splendor(num_envs=1, num_players=num_players, seed=seed,
                   max_turns=max_turns, report_interval=10**9)
    obs, _ = env.reset(seed=seed)
    for a in agents:
        a.reset()
    actions = np.full(num_players, L.PASS, dtype=np.int32)
    for _ in range(12 * env.max_turns):
        sys.stdout.flush()   # the C renderer writes straight to stdout
        env.render()
        mask = L.legal_mask(obs, num_players)
        to_move = L.to_move(obs, num_players)
        actions[:] = L.PASS
        seat = int(np.argmax(to_move)) if to_move.any() else 0
        actions[seat] = agents[seat].act(obs[seat:seat + 1], mask[seat:seat + 1],
                                         env, np.array([0]))[0]
        obs, rewards, terminals, _, _ = env.step(actions)
        if terminals[0]:     # the env has already dealt a new game
            r = rewards[:num_players]
            print('game over: ' + ('draw' if r.max() < 0.5 else
                                   f'seat {int(np.argmax(r))} wins'))
            break
    env.close()


def evaluate(load_model_path='latest', opponent='random', games=1000,
             num_players=2, temperature=0.0, render=False, device='cpu',
             seed=0, max_turns=None, env_kwargs=None, **unused):
    """Seat 0 = `load_model_path`, the other seats = `opponent`. Prints a report."""
    num_players = (env_kwargs or {}).get('num_players', num_players)
    me = A.make_agent(load_model_path or 'random', num_players, device,
                      seed=seed, temperature=temperature)
    seats = [me] + [A.make_agent(opponent, num_players, device, seed=seed + 1 + s,
                                 temperature=temperature)
                    for s in range(num_players - 1)]
    print(f'seat 0: {me.name}   opponents: {seats[1].name} '
          f'({num_players} players)')

    if render:
        render_game(seats, num_players, seed, max_turns)
        return {}

    res = A.play_games(seats, games, num_players, seed=seed, max_turns=max_turns)
    ok = res['finished']
    n = max(int(ok.sum()), 1)
    winner = res['winner'][ok]
    win = float((winner == 0).mean())
    draw = float((winner == -1).mean())
    loss = 1.0 - win - draw
    points = res['points'][ok].mean(0)
    print(f'games:       {int(ok.sum())} / {games} finished')
    print(f'win rate:    {win:.3f}')
    print(f'draw rate:   {draw:.3f}')
    print(f'loss rate:   {loss:.3f}')
    print('mean points: ' + '  '.join(f'seat {s} {p:5.2f}'
                                      for s, p in enumerate(points)))
    print(f'mean turns:  {res["turns"][ok].mean():.1f}')
    return dict(win=win, draw=draw, loss=loss, points=points.tolist(),
                turns=float(res['turns'][ok].mean()), games=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(description='Splendor policy evaluation')
    ap.add_argument('--load-model-path', type=str, default='latest',
                    help="checkpoint path, 'latest', 'random' or 'greedy'")
    ap.add_argument('--opponent', type=str, default='random',
                    help="'random', 'greedy', 'latest' or a checkpoint path")
    ap.add_argument('--games', type=int, default=1000)
    ap.add_argument('--num-players', type=int, default=2)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max-turns', type=int, default=None)
    ap.add_argument('--render', action='store_true',
                    help='play one game and print the board every turn')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    evaluate(args.load_model_path, opponent=args.opponent, games=args.games,
             num_players=args.num_players, temperature=args.temperature,
             render=args.render, device=args.device, seed=args.seed,
             max_turns=args.max_turns)


if __name__ == '__main__':
    main()
