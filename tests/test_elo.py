"""Tests for the evaluation stack: GreedyAgent, Bradley-Terry, elo.py CLI."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import elo
from splendor import agents as A
from splendor import layout as L
from splendor.splendor import Splendor


def latest_checkpoint():
    import glob
    hits = glob.glob(os.path.join(ROOT, 'experiments', '**', '*.pt'),
                     recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None


# --- GreedyAgent ------------------------------------------------------------
@pytest.mark.parametrize('num_players', [2, 4])
def test_greedy_only_plays_legal_actions(num_players):
    """Over 2000 random-play steps, every greedy action is legal in every seat."""
    P = num_players
    env = Splendor(num_envs=32, num_players=P, seed=7, report_interval=10**9)
    obs, _ = env.reset(seed=7)
    greedy = A.GreedyAgent()
    rng = np.random.default_rng(7)
    rows_checked = 0
    for _ in range(2000 // 32 * P):
        mask = L.legal_mask(obs, P)
        rows = np.flatnonzero(L.to_move(obs, P))
        if len(rows):
            actions = greedy.act(obs[rows], mask[rows])
            assert actions.dtype == np.int32
            assert mask[rows, actions].all(), 'greedy returned an illegal action'
            rows_checked += len(rows)
        obs = env.step(L.random_legal_actions(obs, P, rng))[0]
    assert rows_checked > 2000
    env.close()


def test_greedy_beats_random():
    """Greedy wins well over 70% of 300 two-player games, both seat orders."""
    wins = 0
    for seat, agents in enumerate(([A.GreedyAgent(), A.RandomAgent(1)],
                                   [A.RandomAgent(2), A.GreedyAgent()])):
        res = A.play_games(agents, 150, 2, seed=100 + seat)
        assert res['finished'].all()
        wins += int((res['winner'][res['finished']] == seat).sum())
    assert wins / 300 > 0.7, f'greedy win rate {wins / 300:.3f}'


def test_greedy_is_fast():
    """Well under 1 ms per decision row."""
    import time
    env = Splendor(num_envs=64, num_players=2, seed=1, report_interval=10**9)
    obs, _ = env.reset(seed=1)
    greedy, rng = A.GreedyAgent(), np.random.default_rng(1)
    for _ in range(5):
        obs = env.step(L.random_legal_actions(obs, 2, rng))[0]
    rows = np.flatnonzero(L.to_move(obs, 2))
    mask = L.legal_mask(obs, 2)
    greedy.act(obs[rows], mask[rows])           # warm up
    t = time.perf_counter()
    for _ in range(10):
        greedy.act(obs[rows], mask[rows])
    per_row = (time.perf_counter() - t) / (10 * len(rows))
    assert per_row < 1e-3, f'{per_row * 1e3:.3f} ms per row'
    env.close()


# --- Bradley-Terry ----------------------------------------------------------
def test_bradley_terry_recovers_ordering():
    """A > B > C in the pairwise counts must come out as A > B > C in Elo."""
    names = ['a', 'b', 'c']
    wins = np.array([[0, 70, 90],
                     [30, 0, 65],
                     [10, 35, 0]], dtype=float)
    draws = np.zeros((3, 3), dtype=float)
    table = elo.ratings(names, wins, draws, anchor='c')
    assert table['a']['elo'] > table['b']['elo'] > table['c']['elo']
    assert table['c']['elo'] == 0.0                 # anchor
    assert table["a"]["games"] == 200 and table["a"]["wins"] == 160
    # a beats b 70-30 -> roughly 150 Elo apart
    assert 80 < table['a']['elo'] - table['b']['elo'] < 260


def test_bradley_terry_handles_perfect_and_winless():
    p = elo.bradley_terry(np.array([[0, 10], [0, 0]], dtype=float))
    assert np.all(np.isfinite(p)) and p[0] > p[1]


def test_draws_are_half_a_win():
    names = ['x', 'y']
    wins = np.zeros((2, 2))
    draws = np.array([[0, 50], [50, 0]], dtype=float)
    table = elo.ratings(names, wins, draws, anchor=None)
    assert abs(table['x']['elo'] - table['y']['elo']) < 1e-6
    assert table['x']['draws'] == 50 and table['x']['games'] == 50


# --- CLIs -------------------------------------------------------------------
def run_cli(args, tmp_path, name):
    out = str(tmp_path / name)
    proc = subprocess.run([sys.executable, 'elo.py', *args, '--out', out],
                          cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    with open(out) as f:
        return json.load(f), proc.stdout


def test_elo_cli_with_a_checkpoint(tmp_path):
    """random / greedy / latest at the player count `latest` was trained for."""
    ckpt = latest_checkpoint()
    if ckpt is None:
        pytest.skip('no checkpoint in experiments/')
    P = elo.checkpoint_players('latest') or 2
    # 3 participants are enough for 2 or 3 seats; a 4th seat needs one more
    extra = ['random'] * max(0, P - 3)
    table, stdout = run_cli(['--num-players', str(P), '--games', '40',
                             '--participants', 'random', 'greedy', 'latest',
                             *extra], tmp_path, 'elo2.json')
    assert 'latest' in table, stdout
    assert set(table) >= {'random', 'greedy', 'latest'}
    assert table['random']['elo'] == 0.0                  # anchored
    assert table['greedy']['elo'] > table['random']['elo'] + 200
    assert table['greedy']['games'] >= 40
    # `latest` is whatever is training right now, so only its bookkeeping is
    # asserted; the rating machinery itself is covered by the tests above.
    assert table['latest']['games'] == table['greedy']['games']
    assert np.isfinite(table['latest']['elo'])
    assert 'win%' in stdout
    # the printed table (after the ---- rule) is sorted by Elo, best first
    lines = stdout.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('-----')) + 1
    printed = []
    for l in lines[start:]:
        if not l.strip():
            break
        printed.append(l.split()[0])
    assert printed == sorted(table, key=lambda n: -table[n]['elo'])
    assert printed[0] != 'random'   # a trained or heuristic agent leads


def test_elo_cli_three_players(tmp_path):
    # a 2-player checkpoint has the wrong obs size for 3 players, so the third
    # participant is a second random agent
    table, stdout = run_cli(['--num-players', '3', '--games', '30',
                             '--participants', 'random', 'greedy', 'random'],
                            tmp_path, 'elo3.json')
    assert set(table) == {'random', 'greedy', 'random#2'}
    assert table['greedy']['elo'] > table['random']['elo'] + 200
    assert table['greedy']['elo'] > table['random#2']['elo'] + 200
    assert table['greedy']['games'] == 60


def test_evaluate_reports_rates():
    import evaluate
    res = evaluate.evaluate('greedy', opponent='random', games=40,
                            num_players=2, seed=11)
    assert res['games'] == 40
    assert res['win'] > 0.7
    assert abs(res['win'] + res['draw'] + res['loss'] - 1.0) < 1e-9
    assert len(res['points']) == 2 and res['turns'] > 0


def test_greedy_handles_discard_and_noble_phases():
    """Greedy self-play reaches both sub-phases and stays legal there."""
    P = 3
    env = Splendor(num_envs=48, num_players=P, seed=3, report_interval=10**9)
    obs, _ = env.reset(seed=3)
    greedy = [A.GreedyAgent(seed=s) for s in range(P)]
    actions = np.full(env.num_agents, L.PASS, dtype=np.int32)
    discards = nobles = 0
    for _ in range(300):
        mask = L.legal_mask(obs, P)
        to_move = L.to_move(obs, P).reshape(-1, P)
        actions[:] = L.PASS
        for s in range(P):
            games = np.flatnonzero(to_move[:, s])
            if not len(games):
                continue
            rows = games * P + s
            a = greedy[s].act(obs[rows], mask[rows])
            assert mask[rows, a].all()
            actions[rows] = a
            discards += int(mask[rows, L.DISCARD:L.DISCARD + 6].any(1).sum())
            nobles += int(mask[rows, L.CHOOSE_NOBLE:
                                L.CHOOSE_NOBLE + L.NOBLE_MAX].any(1).sum())
        obs = env.step(actions)[0]
    assert discards > 0 and nobles > 0
    env.close()
