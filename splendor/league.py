"""League self-play environment.

`League` wraps an inner `Splendor` vector env and exposes exactly ONE agent per
game: the learner, which always sits at seat 0.  The other seats are played
inside the env by frozen opponents (the current `latest.pt`, members of a
checkpoint pool, or uniform-random-legal play when no checkpoint exists yet).

One wrapper step = "advance every game until it is the learner's turn again":
the learner's action is applied to the seat-0 turn, then opponents keep acting
(possibly through discard / noble sub-phases) until the env waits on seat 0.
The wrapper reward is the sum of the learner's inner rewards over those
sub-steps and the wrapper terminal is their OR.

Opponents are resampled per game whenever a game ends: with probability
`latest_frac` a seat gets the latest model, otherwise a pool member drawn with
PFSP weights `(1 - w_i)**pfsp_exponent + 0.05`, where `w_i` is an EMA of the
learner's win rate against member i (so the learner plays the checkpoints that
beat it more often).

Everything runs on CPU inside the worker process: `torch.set_num_threads(1)`,
`torch.inference_mode()`, one batched forward per distinct model per sub-step.
"""
import glob
import os

import numpy as np
import gymnasium

import pufferlib

from splendor import binding
from splendor import layout as L
from splendor.splendor import Splendor

# Opponent model ids stored in `League.opp`.  >= 0 indexes `League.pool_paths`.
RANDOM = -2      # uniform random over the legal mask
LATEST = -1      # the model currently loaded from `latest_path`


class League(pufferlib.PufferEnv):
    """Self-play league env; `num_agents == num_envs` (the learner is seat 0).

    Construction is cheap and never touches the filesystem: models are loaded
    lazily on the first `reset`, so the driver env built by
    `pufferlib.vector.make` in the main process costs nothing and `latest.pt`
    does not have to exist.
    """

    def __init__(self, num_envs=512, num_players=2,
                 pool_dir='experiments/pool',
                 latest_path='experiments/latest.pt',
                 latest_frac=0.5, pfsp_exponent=2.0, reload_interval=500,
                 opponent_temperature=1.0, wr_ema=0.99,
                 max_turns=None, reward_point=0.02, reward_card=0.0,
                 reward_win=1.0, reward_loss=-1.0, report_interval=128,
                 render_mode=None, buf=None, seed=0):
        if not 2 <= num_players <= 4:
            raise ValueError('num_players must be 2, 3 or 4')

        self.num_envs = num_envs
        self.num_players = num_players
        self.pool_dir = pool_dir
        self.latest_path = latest_path
        self.latest_frac = float(latest_frac)
        self.pfsp_exponent = float(pfsp_exponent)
        self.reload_interval = int(reload_interval)
        self.opponent_temperature = float(opponent_temperature)
        self.wr_ema = float(wr_ema)
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.obs_n = L.obs_size(num_players)

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_n,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(L.NUM_ACTIONS)
        self.num_agents = num_envs

        super().__init__(buf=buf)

        self.env = Splendor(num_envs=num_envs, num_players=num_players,
                            max_turns=max_turns, reward_point=reward_point,
                            reward_card=reward_card, reward_win=reward_win,
                            reward_loss=reward_loss, report_interval=10**9,
                            render_mode=render_mode, seed=seed)
        self.max_turns = self.env.max_turns

        p = num_players
        # Views on the inner buffers: (num_envs, P, ...) instead of (num_envs*P, ...)
        self._obs3 = self.env.observations.reshape(num_envs, p, self.obs_n)
        self._rew2 = self.env.rewards.reshape(num_envs, p)
        self._term2 = self.env.terminals.reshape(num_envs, p)
        self._inner_actions = self.env.actions        # written in place, no copy
        self._tm_off = L.to_move_offset(p)
        self._game0 = np.arange(num_envs) * p         # inner row of every seat 0
        self._max_iters = 8 * p + 8

        self.rng = np.random.default_rng(int(seed or 0) * 1000003 + 20260911)
        self.opp = np.full((num_envs, p), RANDOM, dtype=np.int64)
        self.tick = 0
        self._loaded = False

        # Model registry.  `latest` is keyed by sentinel, pool members by path.
        self._latest_model = None
        self._latest_mtime = None
        self._pool_models = {}
        self.pool_paths = []
        self.pool_wr = np.zeros(0, dtype=np.float64)
        self._wr_memo = {}
        self.wr_latest = 0.5
        self.wr_pool = 0.5
        self._n_latest = 0
        self._n_pool = 0
        self._torch = None

    # --- model management ---------------------------------------------------
    def _torch_mod(self):
        if self._torch is None:
            import torch
            torch.set_num_threads(1)   # opponents run on one core per worker
            self._torch = torch
        return self._torch

    def _load(self, path):
        """Load a checkpoint as a CPU eval policy, or None if it is unusable."""
        self._torch_mod()
        try:
            from splendor import agents
            return agents.load_policy(path, self.num_players, 'cpu')
        except Exception:
            # A truncated / mid-write file or a checkpoint for a different
            # player count: fall back to random play rather than crashing.
            return None

    def _refresh(self):
        """Re-read latest.pt (if its mtime moved) and re-list the pool."""
        try:
            mtime = os.path.getmtime(self.latest_path)
        except OSError:
            mtime = None
        if mtime != self._latest_mtime:
            self._latest_mtime = mtime
            self._latest_model = self._load(self.latest_path) if mtime else None

        paths = sorted(glob.glob(os.path.join(self.pool_dir, '*.pt'))) \
            if self.pool_dir else []
        if paths != self.pool_paths:
            appended = paths[:len(self.pool_paths)] == self.pool_paths
            self._pool_models = {k: v for k, v in self._pool_models.items()
                                 if k in set(paths)}
            self.pool_paths = paths
            self.pool_wr = np.array([self._wr_memo.get(p, 0.5) for p in paths],
                                    dtype=np.float64)
            if not appended:
                # Indices shifted; every stored pool id is now meaningless.
                self._assign(np.arange(self.num_envs))
        self._loaded = True

    def _model(self, model_id):
        if model_id == LATEST:
            return self._latest_model
        if 0 <= model_id < len(self.pool_paths):
            path = self.pool_paths[model_id]
            if path not in self._pool_models:      # lazy: load on first use
                self._pool_models[path] = self._load(path)
            return self._pool_models[path]
        return None

    # --- opponent sampling --------------------------------------------------
    def _assign(self, games):
        """(Re)draw the opponents of `games` for seats 1 .. P-1."""
        games = np.asarray(games)
        n = games.size
        if n == 0:
            return
        k = self.num_players - 1
        npool = len(self.pool_paths)
        if npool:
            w = np.maximum(1.0 - self.pool_wr, 0.0) ** self.pfsp_exponent + 0.05
            w /= w.sum()
            ids = self.rng.choice(npool, size=(n, k), p=w)
            ids = np.where(self.rng.random((n, k)) < self.latest_frac, LATEST, ids)
        else:
            ids = np.full((n, k), LATEST, dtype=np.int64)
        if self._latest_model is None:
            ids = np.where(ids == LATEST, RANDOM, ids)
        self.opp[games[:, None], np.arange(1, self.num_players)[None, :]] = ids

    def _record(self, finished):
        """Update the win-rate EMAs from the learner's terminal rewards."""
        r = self._rew2[finished, 0]
        # +-1 win/loss dominates the <= 0.16 point shaping, so 0.5 separates them
        score = np.where(r > 0.5, 1.0, np.where(r < -0.5, 0.0, 0.5))
        ids = self.opp[finished, 1:].ravel()
        scores = np.repeat(score, self.num_players - 1)
        a = self.wr_ema
        for m in np.unique(ids):
            sel = ids == m
            k = int(sel.sum())
            s = float(scores[sel].mean())
            f = a ** k                 # k EMA updates at once
            if m == LATEST:
                self.wr_latest = self.wr_latest * f + s * (1.0 - f)
                self._n_latest += k
            elif 0 <= m < len(self.pool_wr):
                self.pool_wr[m] = self.pool_wr[m] * f + s * (1.0 - f)
                self._wr_memo[self.pool_paths[m]] = float(self.pool_wr[m])
                self.wr_pool = self.wr_pool * f + s * (1.0 - f)
                self._n_pool += k
        self._assign(finished)

    # --- opponent inference -------------------------------------------------
    def _opponent_actions(self, model_id, rows):
        obs = self.env.observations[rows]
        net = self._model(model_id)
        if net is None:
            return L.random_legal_actions(obs, self.num_players, self.rng)
        torch = self._torch
        with torch.inference_mode():
            logits = net.forward_eval(torch.from_numpy(obs), {})[0].float()
            t = self.opponent_temperature
            if t > 0:   # Gumbel-max == sampling, without a softmax/multinomial
                u = torch.rand_like(logits).clamp_(1e-7, 1.0 - 1e-7)
                logits = logits / t - torch.log(-torch.log(u))
            a = logits.argmax(dim=1)
        return a.numpy().astype(np.int32)

    # --- the inner loop -----------------------------------------------------
    def _advance(self, actions, consumed):
        """Step the inner env until every game waits on seat 0 (or the cap)."""
        p = self.num_players
        obs3, ia = self._obs3, self._inner_actions
        for _ in range(self._max_iters):
            to_move = obs3[:, :, self._tm_off] > 0          # (num_envs, P)
            at_learner = to_move[:, 0]
            need = at_learner & ~consumed
            opp = to_move.any(axis=1) & ~at_learner
            any_opp = opp.any()
            if not need.any() and not any_opp:
                break

            ia[:] = L.NOOP                  # every other game is frozen
            games = np.flatnonzero(need)
            if games.size:
                ia[self._game0[games]] = actions[games]
                consumed[games] = True
            if any_opp:
                og = np.flatnonzero(opp)
                seats = to_move[og].argmax(axis=1)
                rows = og * p + seats
                mids = self.opp[og, seats]
                for m in np.unique(mids):
                    sel = np.flatnonzero(mids == m)
                    ia[rows[sel]] = self._opponent_actions(m, rows[sel])

            self.env.step(ia)
            self.rewards += self._rew2[:, 0]
            term = self._term2[:, 0]
            if term.any():
                self.terminals |= term
                self._record(np.flatnonzero(term))

    # --- PufferEnv API ------------------------------------------------------
    def reset(self, seed=None):
        self.tick = 0
        self.env.reset(seed)
        self._refresh()
        self._assign(np.arange(self.num_envs))
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        # Play the opponents' opening moves so the learner's first observation
        # is one it actually has to act on (the start seat is randomized).
        self._advance(None, np.ones(self.num_envs, dtype=bool))
        self.rewards[:] = 0
        self.observations[:] = self._obs3[:, 0]
        return self.observations, []

    def step(self, actions):
        if not self._loaded:
            self._refresh()
        self.tick += 1
        if self.reload_interval and self.tick % self.reload_interval == 0:
            self._refresh()

        actions = np.asarray(actions).reshape(-1).astype(np.int32, copy=False)
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        # Games the learner was not asked to move in get their action dropped:
        # the observation it answered said "not your turn" (pass-only mask).
        self._advance(actions, ~(self._obs3[:, 0, self._tm_off] > 0))
        self.observations[:] = self._obs3[:, 0]

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.env.c_envs) or {}
            log.update(self.league_stats())
            if log:
                info.append(log)
        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def league_stats(self):
        stats = {'pool_size': len(self.pool_paths)}
        if self._n_latest:
            stats['wr_latest'] = float(self.wr_latest)
        if self._n_pool:
            stats['wr_pool'] = float(self.wr_pool)
        return stats

    def render(self, env_index=0):
        return self.env.render(env_index)

    def close(self):
        self.env.close()


def test_performance(num_envs=512, num_players=2, timeout=10.0,
                     latest_path='experiments/latest.pt', pool_dir='',
                     latest_frac=1.0, seed=0):
    """Wrapper steps/s with random learner actions (see README / DESIGN)."""
    import time
    env = League(num_envs=num_envs, num_players=num_players,
                 latest_path=latest_path, pool_dir=pool_dir,
                 latest_frac=latest_frac, report_interval=10**9, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(0)
    have = env._latest_model is not None
    env.step(L.random_legal_actions(obs, num_players, rng))  # warm up

    tick, start = 0, time.time()
    while time.time() - start < timeout:
        obs = env.step(L.random_legal_actions(obs, num_players, rng))[0]
        tick += 1
    elapsed = time.time() - start
    print(f'League num_envs={num_envs} players={num_players} '
          f'latest={"loaded" if have else "random"} steps={tick}')
    print(f'  wrapper steps/s: {num_envs*tick/elapsed:,.0f}')
    print(f'  calls/s:         {tick/elapsed:,.1f}')
    env.close()
    return num_envs * tick / elapsed


if __name__ == '__main__':
    test_performance()
