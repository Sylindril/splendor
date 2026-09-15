"""Pure-python constants for the Splendor observation and action layout.

Shared by the C header (which must match these numbers), the policy, the
tests and evaluate.py. No dependency on the compiled extension.
"""
import numpy as np

# --- Colors -----------------------------------------------------------------
NUM_COLORS = 5          # white, blue, green, red, black
GOLD = 5                # index of the joker in a 6-long token vector
COLOR_NAMES = ('white', 'blue', 'green', 'red', 'black', 'gold')

# --- Card / player block sizes ---------------------------------------------
CARD_N = 11             # cost[5] + bonus one-hot[5] + points[1]
RESERVED_N = CARD_N + 1  # card + present flag
PLAYER_N = 48           # tokens[6] + bonuses[5] + points[1] + 3 * RESERVED_N
NOBLE_MAX = 5           # max nobles dealt (num_players + 1, P <= 4)
NUM_FACEUP = 12         # 3 tiers x 4 slots
NUM_TIERS = 3
DECK_SIZES = (40, 30, 20)

# --- Observation offsets ----------------------------------------------------
BANK = 0                                   # 6 bytes
DECKS = BANK + 6                           # 3 bytes
FACEUP = DECKS + 3                         # 12 * CARD_N = 132 bytes
NOBLES = FACEUP + NUM_FACEUP * CARD_N      # NOBLE_MAX * 5 = 25 bytes
PLAYERS = NOBLES + NOBLE_MAX * NUM_COLORS  # 166; PLAYER_N bytes per player

# Offsets inside a player block
P_TOKENS = 0
P_BONUS = 6
P_POINTS = 11
P_RESERVED = 12


def obs_size(num_players):
    """Total observation length for a game with `num_players` seats."""
    return PLAYERS + PLAYER_N * num_players + NUM_ACTIONS + 2


def num_players_from_obs(obs_n):
    """Inverse of obs_size."""
    return (obs_n - PLAYERS - NUM_ACTIONS - 2) // PLAYER_N


def player_offset(index):
    """Byte offset of the `index`-th player block (0 = me, perspective-relative)."""
    return PLAYERS + PLAYER_N * index


def mask_offset(num_players):
    return PLAYERS + PLAYER_N * num_players


def turn_offset(num_players):
    return mask_offset(num_players) + NUM_ACTIONS


def to_move_offset(num_players):
    """Byte that is 1 iff it is this seat's move (even when it can only pass)."""
    return turn_offset(num_players) + 1


def to_move(obs, num_players):
    """Bool per row: this seat is the one the env is waiting on."""
    return _as_2d(obs)[:, to_move_offset(num_players)] > 0


# --- Actions ----------------------------------------------------------------
TAKE3 = 0            # 10 combos of 3 distinct colors
TAKE2 = 10           # 10 combos of 2 distinct colors
TAKE1 = 20           # 5 single colors
TAKE2SAME = 25       # 5 colors, 2 of the same
RESERVE_FACEUP = 30  # 12 slots (tier * 4 + pos)
RESERVE_DECK = 42    # 3 deck tops
BUY_FACEUP = 45      # 12 slots
BUY_RESERVED = 57    # 3 own reserved cards
PASS = 60            # legal only when no other action is (or the seat is idle)
DISCARD = 61         # 6: return one token of color c (incl. gold) while over 10 tokens
CHOOSE_NOBLE = 67    # 5: choose dealt noble i when several qualify after a buy
NUM_ACTIONS = 72
NOOP = -1            # not an action: freezes the game for this step (see Splendor.put_state)

COMBOS3 = [(a, b, c)
           for a in range(NUM_COLORS)
           for b in range(a + 1, NUM_COLORS)
           for c in range(b + 1, NUM_COLORS)]
COMBOS2 = [(a, b)
           for a in range(NUM_COLORS)
           for b in range(a + 1, NUM_COLORS)]
assert len(COMBOS3) == 10 and len(COMBOS2) == 10

# Starting bank per color by player count (gold is always 5)
BANK_PER_COLOR = {2: 4, 3: 5, 4: 7}
GOLD_TOKENS = 5


def initial_tokens(num_players):
    """Per-color token totals in the game, including gold."""
    n = BANK_PER_COLOR[num_players]
    return np.array([n] * NUM_COLORS + [GOLD_TOKENS], dtype=np.int64)


# --- Helpers ----------------------------------------------------------------
def _as_2d(obs):
    obs = np.asarray(obs)
    return obs.reshape(1, -1) if obs.ndim == 1 else obs


def legal_mask(obs, num_players):
    """Bool array (n_agents, NUM_ACTIONS) of legal actions from raw observations."""
    obs = _as_2d(obs)
    off = mask_offset(num_players)
    return obs[:, off:off + NUM_ACTIONS] > 0


def random_legal_actions(obs, num_players, rng):
    """Uniform random legal action per row. Falls back to PASS if nothing is legal."""
    mask = legal_mask(obs, num_players)
    cum = np.cumsum(mask, axis=1)
    total = cum[:, -1:]
    pick = rng.random((mask.shape[0], 1)) * np.maximum(total, 1)
    actions = (cum <= pick).sum(axis=1)
    actions = np.where(total[:, 0] > 0, actions, PASS)
    return np.minimum(actions, NUM_ACTIONS - 1).astype(np.int32)


def acting(mask):
    """Bool per row: the seat has some legal action other than PASS
    (i.e. it is the seat to move, possibly in a discard / noble sub-phase)."""
    mask = np.asarray(mask)
    return mask[..., :PASS].any(axis=-1) | mask[..., PASS + 1:].any(axis=-1)


def obs_scale(num_players):
    """Per-index multiplier putting every observation byte roughly in [0, 1]."""
    n = obs_size(num_players)
    s = np.ones(n, dtype=np.float32)
    s[BANK:BANK + 6] = 1 / 7
    s[DECKS:DECKS + 3] = 1 / 40
    for i in range(NUM_FACEUP):
        o = FACEUP + CARD_N * i
        s[o:o + 5] = 1 / 7          # cost
        s[o + 5:o + 10] = 1.0       # bonus one-hot
        s[o + 10] = 1 / 5           # points
    s[NOBLES:NOBLES + NOBLE_MAX * NUM_COLORS] = 1 / 4
    for p in range(num_players):
        o = player_offset(p)
        s[o + P_TOKENS:o + P_TOKENS + 6] = 1 / 10
        s[o + P_BONUS:o + P_BONUS + 5] = 1 / 10
        s[o + P_POINTS] = 1 / 20
        for r in range(3):
            c = o + P_RESERVED + RESERVED_N * r
            s[c:c + 5] = 1 / 7
            s[c + 5:c + 10] = 1.0
            s[c + 10] = 1 / 5
            s[c + 11] = 1.0         # present flag
    s[mask_offset(num_players):turn_offset(num_players)] = 1.0
    s[turn_offset(num_players)] = 1 / 255
    s[to_move_offset(num_players)] = 1.0
    return s
