"""Tests for the batched PUCT search and the AlphaZero loop.

    ~/miniforge3/envs/splendor/bin/python -m pytest tests/test_mcts.py -q

Everything runs against the newest checkpoint under experiments/ ('latest');
the whole file is skipped when the C extension or a checkpoint is missing.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from splendor import layout as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from splendor.splendor import Splendor
    from splendor import agents as A
    from splendor.mcts import MCTS, MCTSAgent
    from alphazero import matching_checkpoints
    IMPORT_ERROR = None
except Exception as exc:                                  # pragma: no cover
    Splendor = None
    IMPORT_ERROR = exc

needs_ext = pytest.mark.skipif(
    Splendor is None, reason=f'splendor not importable: {IMPORT_ERROR}')


@pytest.fixture(scope='module')
def net2():
    """A trained 2-player net.

    experiments/ is shared with live training runs, so the newest checkpoint can
    belong to another table size or to a run that started seconds ago. Walk the
    2-player checkpoints newest first and take the first one whose greedy policy
    already beats a random opponent; MCTS strength is only meaningful on top of
    a net that has learned something.
    """
    os.chdir(ROOT)
    for path in matching_checkpoints(2)[:8]:
        net = A.load_policy(path, 2, 'cpu')
        r = A.play_games([A.PolicyAgent(net, 'cpu'), A.RandomAgent(1)],
                         40, 2, seed=17)
        if A.win_rates(r, 2)[0] >= 0.7:
            return net
    pytest.skip('no trained 2-player checkpoint under experiments/')


def rollout(env, steps, seed=0):
    """Advance every game a few random-legal turns to get varied root states."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        env.step(L.random_legal_actions(env.observations, env.num_players, rng))


def seats_of(env):
    P = env.num_players
    return env.observations[:, L.to_move_offset(P)].reshape(-1, P).argmax(axis=1)


# --- 1: legality and visit bookkeeping --------------------------------------
@needs_ext
def test_search_legal_and_visit_counts(net2):
    B, SIMS = 8, 32
    env = Splendor(num_envs=B, num_players=2, seed=1, report_interval=10**9)
    env.reset(seed=1)
    rollout(env, 25, seed=1)
    states = [env.get_state(i) for i in range(B)]
    seats = seats_of(env)
    mask = L.legal_mask(env.observations, 2).reshape(B, 2, -1)[np.arange(B), seats]

    mcts = MCTS(net2, 2, sims=SIMS, device='cpu', capacity=B, noise=True, seed=0)
    visits, values = mcts.search(states, seats)

    assert visits.shape == (B, L.NUM_ACTIONS)
    assert values.shape == (B, 2)
    # every simulation lands on exactly one root child
    assert (visits.sum(axis=1) == SIMS).all()
    # nothing illegal is ever proposed or even visited
    assert not (visits > 0)[~mask].any()
    for b in range(B):
        assert mask[b][int(visits[b].argmax())]
    assert np.isfinite(values).all() and (np.abs(values) <= 1.0 + 1e-5).all()
    # the root eval plus one simulation each
    assert (mcts.N[:B, 0] == SIMS + 1).all()
    mcts.close()
    env.close()


# --- 2: a position with an affordable points card ---------------------------
@needs_ext
def test_buyable_points_card(net2):
    env = Splendor(num_envs=1, num_players=2, seed=7, report_interval=10**9)
    env.reset(seed=7)
    seat = int(seats_of(env)[0])
    # make every face-up card affordable for the seat to move
    bonuses = [0] * 10
    bonuses[seat * 5:seat * 5 + 5] = [7] * 5
    env.put_state(0, bonuses=bonuses)

    obs = env.observations[seat]
    mask = obs[L.mask_offset(2):L.turn_offset(2)] > 0
    buys = [a for a in range(L.BUY_FACEUP, L.BUY_FACEUP + 12) if mask[a]]
    points = [int(obs[L.FACEUP + L.CARD_N * (a - L.BUY_FACEUP) + 10]) for a in buys]
    assert buys and max(points) > 0, 'expected an affordable scoring card'

    mcts = MCTS(net2, 2, sims=64, device='cpu', capacity=1, noise=False, seed=0)
    visits, values = mcts.search([env.get_state(0)], [seat])
    action = int(visits[0].argmax())
    assert mask[action], 'MCTS proposed an illegal action'
    v = float(values[0, seat])
    assert np.isfinite(v) and -1.0 <= v <= 1.0
    mcts.close()
    env.close()


# --- 3: strength against a random opponent ----------------------------------
@needs_ext
def test_mcts_beats_random(net2):
    games = 60
    mcts = MCTSAgent(net2, 2, sims=32, device='cpu', capacity=games, seed=0)
    result = A.play_games([mcts, A.RandomAgent(seed=1)], games, 2, seed=5)
    wr = A.win_rates(result, 2)
    assert result['finished'].all()
    assert wr[0] > 0.8, f'MCTS(32) win rate vs random = {wr[0]:.2f}'
    mcts.close()


@needs_ext
def test_three_players():
    """3 seats work end to end (no 3-player checkpoint exists, so use a fresh net)."""
    from splendor import policy as Pol
    games = 20
    net3 = Pol.Big(A._Spaces(3))
    mcts = MCTSAgent(net3, 3, sims=16, device='cpu', capacity=games, seed=0)
    result = A.play_games([mcts, A.RandomAgent(2), A.RandomAgent(3)],
                          games, 3, seed=11)
    assert result['finished'].all()
    assert (result['turns'] > 0).all()
    mcts.close()


# --- 4: determinization on and off ------------------------------------------
@needs_ext
@pytest.mark.parametrize('determinize', [True, False])
def test_determinize_runs(net2, determinize):
    B = 4
    env = Splendor(num_envs=B, num_players=2, seed=2, report_interval=10**9)
    env.reset(seed=2)
    rollout(env, 40, seed=2)
    states = [env.get_state(i) for i in range(B)]
    seats = seats_of(env)
    mcts = MCTS(net2, 2, sims=16, device='cpu', capacity=B, seed=0,
                determinize=determinize)
    visits, values = mcts.search(states, seats)
    assert (visits.sum(axis=1) == 16).all()
    assert np.isfinite(values).all()
    # the real env must be untouched by the search
    assert all(env.get_state(i) == states[i] for i in range(B))
    mcts.close()
    env.close()


# --- 5: the AlphaZero CLI ---------------------------------------------------
@needs_ext
def test_alphazero_cli():
    out = os.path.join(ROOT, 'experiments', 'az')
    assert not os.path.exists(out), 'experiments/az already exists; refusing to clobber'
    try:
        proc = subprocess.run(
            [sys.executable, 'alphazero.py', '--games', '8', '--sims', '16',
             '--iters', '1', '--epochs', '1'],
            cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
        ckpt = os.path.join(out, 'iter_0.pt')
        assert os.path.exists(ckpt), proc.stdout[-2000:]
        os.chdir(ROOT)
        net = A.load_policy(ckpt, 2, 'cpu')
        assert net.hidden_size == 512
    finally:
        shutil.rmtree(out, ignore_errors=True)
