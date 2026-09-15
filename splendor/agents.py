"""Agents that play Splendor from observations, and a batched match runner.

An Agent maps a batch of observations (one per game where it is that agent's
turn) to actions. Everything needed by a plain policy is inside the observation
(board, players, legal-action mask); search agents additionally get the env
and the game indices so they can snapshot / restore states.
"""
import glob
import os

import numpy as np

from splendor import layout as L
from splendor.splendor import Splendor


class Agent:
    name = 'agent'

    def act(self, obs, mask, env=None, games=None):
        """obs: (n, OBS_N) uint8, mask: (n, NUM_ACTIONS) bool -> (n,) int32 actions.
        env / games: the Splendor env and the game indices these rows belong
        to (only needed by search agents)."""
        raise NotImplementedError

    def reset(self, games=None):
        """Called when the given games (or all) start a new game."""


class RandomAgent(Agent):
    name = 'random'

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, mask, env=None, games=None):
        return L.random_legal_actions(obs, L.num_players_from_obs(obs.shape[1]), self.rng)


def load_policy(path, num_players, device='cpu'):
    """Build the right Policy/Big architecture for a checkpoint and load it."""
    import torch
    from splendor import policy as P
    if path == 'latest':
        path = latest_checkpoint(num_players)
    sd = torch.load(path, map_location=device, weights_only=False)
    while isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    hidden, layers, norm = P.arch_from_state_dict(sd)
    env_like = _Spaces(num_players)
    net = P.Policy(env_like, hidden_size=hidden, layers=layers, norm=norm)
    net.load_state_dict(sd)
    return net.to(device).eval()


def latest_checkpoint(num_players, pattern='experiments/**/*.pt'):
    """Newest checkpoint under experiments/ trained for `num_players` (the
    observation width differs per player count)."""
    import torch
    want = L.obs_size(num_players)
    for path in sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True):
        if os.path.basename(path) == 'trainer_state.pt':
            continue
        try:
            sd = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(sd, dict) and sd.get('encoder.0.weight') is not None \
                    and sd['encoder.0.weight'].shape[1] == want:
                return path
        except Exception:
            continue
    raise FileNotFoundError(f'no checkpoint for {num_players} players under experiments/')


class _Spaces:
    """Just enough of an env for Policy.__init__ (no C extension needed)."""

    def __init__(self, num_players):
        import gymnasium
        self.single_observation_space = gymnasium.spaces.Box(
            0, 255, (L.obs_size(num_players),), np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(L.NUM_ACTIONS)


class PolicyAgent(Agent):
    """A (masked) policy network. Greedy by default; `temperature` > 0 samples."""

    def __init__(self, policy, device='cpu', temperature=0.0, name=None, seed=0):
        import torch
        self.torch = torch
        self.policy = policy.to(device).eval()
        self.device = device
        self.temperature = temperature
        self.name = name or 'policy'
        self.gen = torch.Generator(device='cpu').manual_seed(seed)

    @classmethod
    def from_checkpoint(cls, path, num_players, device='cpu', **kw):
        return cls(load_policy(path, num_players, device), device,
                   name=kw.pop('name', os.path.basename(path)), **kw)

    def act(self, obs, mask, env=None, games=None):
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(obs), device=self.device)
            logits, _ = self.policy.forward_eval(x, {})
            if self.temperature > 0:
                probs = torch.softmax(logits.float() / self.temperature, dim=1).cpu()
                a = torch.multinomial(probs, 1, generator=self.gen)[:, 0]
            else:
                a = logits.argmax(dim=1).cpu()
        return a.numpy().astype(np.int32)


def play_games(agents, num_games, num_players=None, seed=0, max_turns=None,
               env=None, verbose=False):
    """Play `num_games` games in parallel; seat s is controlled by agents[s].

    Returns a dict of per-game arrays: 'winner' (seat index, -1 for a draw),
    'points' (num_games, P) final points, 'turns', plus 'ranks' (num_games, P)
    with 0 = best (ties share the best rank) for multiplayer ratings.
    Each game is scored the first time it finishes; the start seat is random
    inside the env, so seat order carries no advantage.
    """
    P = num_players or len(agents)
    assert len(agents) == P
    env = env or Splendor(num_envs=num_games, num_players=P, seed=seed,
                          max_turns=max_turns, report_interval=10**9)
    obs, _ = env.reset(seed=seed)
    for a in agents:
        a.reset()
    turn_off, pts_off = L.turn_offset(P), L.player_offset(0) + L.P_POINTS
    done = np.zeros(num_games, dtype=bool)
    winner = np.full(num_games, -1, dtype=np.int64)
    points = np.zeros((num_games, P), dtype=np.int64)
    turns = np.zeros(num_games, dtype=np.int64)
    actions = np.empty(env.num_agents, dtype=np.int32)

    for _ in range(12 * env.max_turns):
        if done.all():
            break
        mask = L.legal_mask(obs, P)
        acting = L.to_move(obs, P).reshape(num_games, P)
        actions[:] = L.PASS  # non-acting seats are ignored; a stalled seat passes
        for s in range(P):
            games = np.flatnonzero(acting[:, s] & ~done)
            if len(games) == 0:
                continue
            rows = games * P + s
            actions[rows] = agents[s].act(obs[rows], mask[rows], env, games)
        # obs aliases the env buffer, so snapshot what the step will overwrite
        pre_all = np.stack([obs[s::P][:, pts_off] for s in range(P)], axis=1).astype(np.int64)
        pre_turn = obs[::P][:, turn_off].astype(np.int64)
        obs, rewards, terminals, _, _ = env.step(actions)
        just = terminals.reshape(num_games, P)[:, 0].astype(bool) & ~done
        if just.any():
            r = rewards.reshape(num_games, P)[just]
            # final points = points before the last step + points gained in it
            gained = np.rint((r - np.clip(np.rint(r), -1, 1)) / 0.02).astype(np.int64)
            points[just] = pre_all[just] + gained
            turns[just] = pre_turn[just] + 1
            w = np.argmax(r, axis=1)
            winner[just] = np.where(r.max(axis=1) > 0.5, w, -1)
            done |= just
            for a in agents:
                a.reset(np.flatnonzero(just))
    # ranks from points, tiebreak fewer cards is unknown here -> use reward sign
    ranks = np.zeros((num_games, P), dtype=np.int64)
    order = np.argsort(-points, axis=1, kind='stable')
    for g in range(num_games):
        r = 0
        for i, s in enumerate(order[g]):
            if i > 0 and points[g, s] < points[g, order[g][i - 1]]:
                r = i
            ranks[g, s] = r
    return dict(winner=winner, points=points, turns=turns, ranks=ranks,
                finished=done)


def win_rates(result, num_players):
    """Fraction of finished games won by each seat (draws count for nobody)."""
    w = result['winner'][result['finished']]
    return np.array([(w == s).mean() if len(w) else 0.0 for s in range(num_players)])


# --- Greedy heuristic -------------------------------------------------------
def _take_deltas():
    """(30, 5) token delta of every take action (indices 0 .. TAKE2SAME+4)."""
    d = np.zeros((L.TAKE2SAME + L.NUM_COLORS, L.NUM_COLORS), dtype=np.int16)
    for i, combo in enumerate(L.COMBOS3):
        d[L.TAKE3 + i, list(combo)] = 1
    for i, combo in enumerate(L.COMBOS2):
        d[L.TAKE2 + i, list(combo)] = 1
    for c in range(L.NUM_COLORS):
        d[L.TAKE1 + c, c] = 1
        d[L.TAKE2SAME + c, c] = 2
    return d


TAKE_DELTA = _take_deltas()          # (NUM_TAKES, 5)
NUM_TAKES = TAKE_DELTA.shape[0]      # 30
TAKE_SIZE = TAKE_DELTA.sum(1)        # tokens gained by each take
# Face-up slot s -> tier; reserved cards have no tier in the observation.
FACEUP_TIER = np.repeat(np.arange(L.NUM_TIERS), 4)
_FAR = 1000                          # shortfall of an empty / absent card


class GreedyAgent(Agent):
    """A fixed heuristic yardstick that plays from the observation only.

    Buy the affordable card worth the most points; else take the gems that
    most reduce the shortfall of the card closest to affordable; else reserve
    that card; else any legal take. Discards whatever the closest card needs
    least. Fully vectorized over the batch.
    """

    name = 'greedy'

    def __init__(self, seed=0, name=None):
        self.rng = np.random.default_rng(seed)
        if name:
            self.name = name

    def act(self, obs, mask, env=None, games=None):
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        n, obs_n = obs.shape
        P = L.num_players_from_obs(obs_n)
        mask = L.legal_mask(obs, P) if mask is None else np.asarray(mask) > 0
        mask = mask.reshape(n, L.NUM_ACTIONS)
        o = obs.astype(np.int16)
        rows = np.arange(n)

        me = L.player_offset(0)
        tokens = o[:, me + L.P_TOKENS:me + L.P_TOKENS + 6]          # (n, 6)
        bonus = o[:, me + L.P_BONUS:me + L.P_BONUS + 5]             # (n, 5)
        gold = tokens[:, L.GOLD]
        held = tokens.sum(1)

        # Candidate cards: the 12 face-up slots then my 3 reserved cards, in
        # the same order as the buy actions (BUY_FACEUP .. BUY_RESERVED + 2).
        faceup = o[:, L.FACEUP:L.FACEUP + L.NUM_FACEUP * L.CARD_N]
        faceup = faceup.reshape(n, L.NUM_FACEUP, L.CARD_N)
        res = o[:, me + L.P_RESERVED:me + L.P_RESERVED + 3 * L.RESERVED_N]
        res = res.reshape(n, 3, L.RESERVED_N)
        cards = np.concatenate([faceup, res[:, :, :L.CARD_N]], axis=1)
        present = cards[:, :, L.NUM_COLORS:2 * L.NUM_COLORS].sum(2) > 0
        cost = cards[:, :, :L.NUM_COLORS].astype(np.int32)          # (n, 15, 5)
        points = cards[:, :, 2 * L.NUM_COLORS].astype(np.int32)     # (n, 15)

        need = np.maximum(cost - bonus[:, None, :], 0)              # after bonuses
        deficit = np.maximum(need - tokens[:, None, :L.NUM_COLORS], 0)
        short = np.maximum(deficit.sum(2) - gold[:, None], 0)
        short = np.where(present, short, _FAR)                      # (n, 15)

        # Closest card: least short, then most points.
        close = np.argmin(short * 64 - points, axis=1)
        need_c = need[rows, close]                                  # (n, 5)
        short_c = short[rows, close]
        # Closest face-up card (only those can be reserved).
        close_up = np.argmin(short[:, :L.NUM_FACEUP] * 64
                             - points[:, :L.NUM_FACEUP], axis=1)

        # --- buy: most points, then higher tier, then cheaper ---------------
        tier = np.empty((n, cards.shape[1]), dtype=np.int32)
        tier[:, :L.NUM_FACEUP] = FACEUP_TIER
        costsum = cost.sum(2)
        tier[:, L.NUM_FACEUP:] = np.clip(  # reserved: no tier in the obs
            (costsum[:, L.NUM_FACEUP:] - 3) // 4, 0, 2)
        buy_ok = mask[:, L.BUY_FACEUP:L.BUY_FACEUP + cards.shape[1]]
        buy_score = points * 10000 + tier * 100 - costsum
        buy = L.BUY_FACEUP + np.argmax(np.where(buy_ok, buy_score, -1 << 30), 1)

        # --- take: the legal take that most reduces the closest shortfall ---
        after = tokens[:, None, :L.NUM_COLORS] + TAKE_DELTA[None]   # (n, 30, 5)
        new_short = np.maximum(
            np.maximum(need_c[:, None, :] - after, 0).sum(2) - gold[:, None], 0)
        gain = short_c[:, None] - new_short                         # (n, 30)
        take_ok = mask[:, :NUM_TAKES]
        score = gain.astype(np.float32)
        two_same = np.zeros((n, NUM_TAKES), dtype=np.float32)       # cover a 2+ hole
        two_same[:, L.TAKE2SAME:L.TAKE2SAME + L.NUM_COLORS] = 0.25 * (need_c -
            tokens[:, :L.NUM_COLORS] >= 2)
        score += two_same
        score += 0.01 * TAKE_SIZE                                   # prefer more gems
        score -= 100.0 * (held[:, None] + TAKE_SIZE[None] > 10)     # avoid discarding
        score = np.where(take_ok, score, -1e9)
        take = np.argmax(score, axis=1)
        take_helps = take_ok[rows, take] & (gain[rows, take] > 0)

        # --- fallbacks ------------------------------------------------------
        first_take = np.argmax(take_ok, axis=1)
        any_take = take_ok.any(1)
        first_any = np.argmax(mask[:, :L.PASS], axis=1)
        any_move = mask[:, :L.PASS].any(1)

        act = np.full(n, L.PASS, dtype=np.int64)
        act = np.where(any_move, first_any, act)
        act = np.where(any_take, first_take, act)
        reserve_ok = mask[rows, L.RESERVE_FACEUP + close_up]
        act = np.where(reserve_ok, L.RESERVE_FACEUP + close_up, act)
        act = np.where(take_helps, take, act)
        act = np.where(buy_ok.any(1), buy, act)

        # --- sub-phases override everything ---------------------------------
        disc_ok = mask[:, L.DISCARD:L.DISCARD + 6]
        if disc_ok.any():
            surplus = tokens[:, :L.NUM_COLORS] - need_c            # least useful
            dscore = np.empty((n, 6), dtype=np.float32)
            dscore[:, :L.NUM_COLORS] = surplus + 0.01 * tokens[:, :L.NUM_COLORS]
            dscore[:, L.GOLD] = -1e5                               # gold last
            dscore = np.where(disc_ok, dscore, -1e9)
            act = np.where(disc_ok.any(1), L.DISCARD + np.argmax(dscore, 1), act)
        noble_ok = mask[:, L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + L.NOBLE_MAX]
        if noble_ok.any():
            act = np.where(noble_ok.any(1),
                           L.CHOOSE_NOBLE + np.argmax(noble_ok, 1), act)
        return act.astype(np.int32)


AGENTS = {'random': RandomAgent, 'greedy': GreedyAgent}


def make_agent(spec, num_players=2, device='cpu', seed=0, temperature=0.0,
               name=None):
    """Build an agent from a string: 'random', 'greedy', 'latest' or a .pt path."""
    if isinstance(spec, Agent):
        return spec
    if spec in AGENTS:
        a = AGENTS[spec](seed=seed)
    else:
        a = PolicyAgent.from_checkpoint(spec, num_players, device,
                                        temperature=temperature, seed=seed)
        a.name = name or ('latest' if spec == 'latest'
                          else os.path.basename(spec).replace('.pt', ''))
    if name:
        a.name = name
    return a
