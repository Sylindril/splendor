"""Tests for the league self-play wrapper (splendor/league.py).

    ~/miniforge3/envs/splendor/bin/python -m pytest tests -q

Everything that writes goes to pytest's tmp_path; the only thing read out of
experiments/ is an existing checkpoint (copied, never modified).
"""
import glob
import os
import shutil

import numpy as np
import pytest

from splendor import layout as L

try:
    from splendor.league import League, LATEST, RANDOM
    IMPORT_ERROR = None
except Exception as exc:  # extension not built yet
    League = None
    IMPORT_ERROR = exc

needs_ext = pytest.mark.skipif(
    League is None, reason=f'splendor.league not importable: {IMPORT_ERROR}')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest_checkpoint():
    """The newest top-level experiments/*.pt, or None."""
    # newest 2-player checkpoint (experiments/ also holds 4-player ones)
    from splendor.agents import latest_checkpoint
    try:
        return latest_checkpoint(2, os.path.join(HERE, 'experiments', '**', '*.pt'))
    except FileNotFoundError:
        return None


needs_ckpt = pytest.mark.skipif(
    newest_checkpoint() is None, reason='no experiments/*.pt checkpoint')


def make(tmp_path, **kw):
    kw.setdefault('num_envs', 64)
    kw.setdefault('pool_dir', str(tmp_path / 'pool'))
    kw.setdefault('latest_path', str(tmp_path / 'latest.pt'))
    kw.setdefault('report_interval', 25)
    return League(**kw)


def rollout(env, steps, seed=0):
    """Random legal learner play; returns (terminals seen, infos, obs checks)."""
    p = env.num_players
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    terminals, infos = 0, []
    for i in range(steps):
        # The wrapper must always hand the learner a seat-0 observation it is
        # actually to move in.
        assert np.array_equal(obs, env.env.observations[0::p]), f'step {i}'
        assert L.to_move(obs, p).all(), f'step {i}'
        actions = L.random_legal_actions(obs, p, rng)
        obs, rew, term, trunc, info = env.step(actions)
        assert np.isfinite(rew).all()
        assert rew.shape == (env.num_envs,) and term.shape == (env.num_envs,)
        terminals += int(term.sum())
        infos.extend(info)
    return terminals, infos


# --- (a) no latest.pt: random opponents -------------------------------------
@needs_ext
def test_no_latest_random_opponents(tmp_path):
    env = make(tmp_path, num_envs=64, num_players=2)
    try:
        assert env._latest_model is None
        assert (env.opp[:, 1:] == RANDOM).all()
        terminals, infos = rollout(env, 300, seed=3)
        assert terminals > 0, 'no game finished in 300 wrapper steps'
        assert len(infos) == 300 // 25
        assert all(isinstance(d, dict) for d in infos)
        assert all('pool_size' in d for d in infos)
        # the inner env's own log still comes through
        assert any('game_length' in d for d in infos)
    finally:
        env.close()


# --- (b) a strong latest.pt beats a random learner --------------------------
@needs_ext
@needs_ckpt
def test_random_learner_loses_to_latest(tmp_path):
    latest = tmp_path / 'latest.pt'
    shutil.copyfile(newest_checkpoint(), latest)
    env = make(tmp_path, num_envs=64, num_players=2, latest_frac=1.0)
    try:
        assert env.reset(seed=1) is not None
        assert env._latest_model is not None, 'latest.pt did not load'
        assert (env.opp[:, 1:] == LATEST).all()
        terminals, infos = rollout(env, 800, seed=5)
        assert terminals > 100
        assert env._n_latest > 100
        assert env.wr_latest < 0.3, f'wr_latest = {env.wr_latest}'
        assert infos[-1]['wr_latest'] < 0.3
    finally:
        env.close()


# --- (c) pool sampling and per-member PFSP win rates ------------------------
@needs_ext
@needs_ckpt
def test_pool_sampling(tmp_path):
    pool = tmp_path / 'pool'
    pool.mkdir()
    for name in ('step_00000000000.pt', 'step_00000000001.pt'):
        shutil.copyfile(newest_checkpoint(), pool / name)

    env = make(tmp_path, num_envs=64, num_players=2, latest_frac=0.0)
    try:
        env.reset(seed=2)
        assert len(env.pool_paths) == 2
        assert env._latest_model is None      # no latest.pt was written
        seen = set(np.unique(env.opp[:, 1:]).tolist())
        assert seen == {0, 1}, seen           # latest_frac=0 -> pool only
        _, infos = rollout(env, 600, seed=7)
        seen |= set(np.unique(env.opp[:, 1:]).tolist())
        assert seen == {0, 1}
        # both members accumulated results and moved off the 0.5 prior
        assert env._n_pool > 100
        assert len(env._wr_memo) == 2
        assert all(abs(v - 0.5) > 0.05 for v in env._wr_memo.values()), env._wr_memo
        assert (np.abs(env.pool_wr - 0.5) > 0.05).all(), env.pool_wr
        assert 'wr_pool' in infos[-1] and infos[-1]['pool_size'] == 2
        # a random learner loses to both copies of a strong checkpoint
        assert env.wr_pool < 0.3, env.wr_pool
    finally:
        env.close()


@needs_ext
@needs_ckpt
def test_pool_pfsp_weights_prefer_strong_members(tmp_path):
    """PFSP oversamples members the learner does badly against."""
    pool = tmp_path / 'pool'
    pool.mkdir()
    for name in ('a.pt', 'b.pt'):
        shutil.copyfile(newest_checkpoint(), pool / name)
    env = make(tmp_path, num_envs=256, num_players=2, latest_frac=0.0)
    try:
        env.reset(seed=0)
        env.pool_wr[:] = [0.9, 0.1]   # member 0 beats us rarely, member 1 often
        env._assign(np.arange(env.num_envs))
        ids = env.opp[:, 1]
        frac0 = float((ids == 0).mean())
        # weights are 0.1**2 + 0.05 = 0.06 vs 0.9**2 + 0.05 = 0.86
        assert 0.02 < frac0 < 0.15, frac0
    finally:
        env.close()


# --- (d) three and four players ---------------------------------------------
@needs_ext
@pytest.mark.parametrize('num_players', [3, 4])
def test_multiplayer(tmp_path, num_players):
    env = make(tmp_path, num_envs=32, num_players=num_players,
               report_interval=50)
    try:
        assert env.observations.shape == (32, L.obs_size(num_players))
        assert env.num_agents == 32
        terminals, infos = rollout(env, 200, seed=num_players)
        assert terminals > 0
        assert len(infos) == 4
    finally:
        env.close()


@needs_ext
@needs_ckpt
@pytest.mark.parametrize('num_players', [3, 4])
def test_multiplayer_with_policy_opponents(tmp_path, num_players):
    """A 2-player checkpoint cannot be loaded for P>2: fall back to random."""
    shutil.copyfile(newest_checkpoint(), tmp_path / 'latest.pt')
    env = make(tmp_path, num_envs=16, num_players=num_players, latest_frac=1.0)
    try:
        env.reset(seed=0)
        assert env._latest_model is None            # shape mismatch -> None
        assert (env.opp[:, 1:] == RANDOM).all()
        rollout(env, 100, seed=1)
    finally:
        env.close()


# --- reloading latest.pt while running --------------------------------------
@needs_ext
@needs_ckpt
def test_latest_reloaded_on_mtime_change(tmp_path):
    env = make(tmp_path, num_envs=32, num_players=2, reload_interval=5,
               latest_frac=1.0)
    try:
        env.reset(seed=0)
        assert env._latest_model is None
        rng = np.random.default_rng(0)
        obs = env.observations
        for _ in range(6):
            obs = env.step(L.random_legal_actions(obs, 2, rng))[0]
        assert env._latest_model is None
        shutil.copyfile(newest_checkpoint(), tmp_path / 'latest.pt')
        for _ in range(10):
            obs = env.step(L.random_legal_actions(obs, 2, rng))[0]
        assert env._latest_model is not None
        # Opponents are only redrawn when a game ends, so games in flight keep
        # playing random until they finish; after that everything is LATEST.
        assert (env.opp[:, 1:] == RANDOM).all()
        for _ in range(200):
            obs = env.step(L.random_legal_actions(obs, 2, rng))[0]
        assert (env.opp[:, 1:] == LATEST).all()
        assert env._n_latest > 0
    finally:
        env.close()


# --- rewards / terminals bookkeeping ----------------------------------------
@needs_ext
def test_reward_is_sum_of_learner_rewards(tmp_path):
    """Wrapper reward == inner seat-0 rewards summed over the sub-steps, and a
    terminal step carries a +-1 (or 0 for a draw) outcome."""
    env = make(tmp_path, num_envs=64, num_players=2)
    try:
        rng = np.random.default_rng(11)
        obs, _ = env.reset(seed=11)
        outcomes = []
        for _ in range(400):
            obs, rew, term, _, _ = env.step(L.random_legal_actions(obs, 2, rng))
            assert (np.abs(rew) < 5).all()
            if term.any():
                outcomes.extend(rew[term].tolist())
        assert outcomes
        # win/loss/draw dominate the <= 0.16 shaping on the terminal step
        assert all(abs(o) > 0.8 or abs(o) < 0.2 for o in outcomes)
    finally:
        env.close()


@needs_ext
def test_noop_freezes_games_the_learner_already_moved_in(tmp_path):
    """Two wrapper steps never give the learner two turns in the same game."""
    env = make(tmp_path, num_envs=32, num_players=2)
    try:
        obs, _ = env.reset(seed=4)
        turn_off = L.turn_offset(2)
        before = obs[:, turn_off].astype(int)
        rng = np.random.default_rng(4)
        obs, _, term, _, _ = env.step(L.random_legal_actions(obs, 2, rng))
        after = obs[:, turn_off].astype(int)
        # one learner turn + one opponent turn per game (a few more if either
        # had to discard); games that ended restarted their turn counter
        delta = (after - before)[~term]
        assert ((delta >= 1) & (delta <= 8)).all(), np.unique(delta)
    finally:
        env.close()
