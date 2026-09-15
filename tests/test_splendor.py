"""Tests for the Splendor PufferLib environment.

    ~/miniforge3/envs/splendor/bin/python -m pytest tests -q
"""
import numpy as np
import pytest

from splendor import layout as L

try:
    from splendor.splendor import Splendor
    IMPORT_ERROR = None
except Exception as exc:  # extension not built yet
    Splendor = None
    IMPORT_ERROR = exc

needs_ext = pytest.mark.skipif(
    Splendor is None, reason=f'splendor.binding not importable: {IMPORT_ERROR}')


# --- obs helpers (work on a single row or a stack of rows) -------------------
def bank(obs):
    return np.asarray(obs)[..., L.BANK:L.BANK + 6].astype(int)


def decks(obs):
    return np.asarray(obs)[..., L.DECKS:L.DECKS + 3].astype(int)


def card(obs, slot):
    o = L.FACEUP + L.CARD_N * slot
    return np.asarray(obs)[..., o:o + L.CARD_N].astype(int)


def player(obs, i):
    o = L.player_offset(i)
    return np.asarray(obs)[..., o:o + L.PLAYER_N].astype(int)


def acting_seat(obs, num_players):
    """Index of the seat with a legal action other than PASS, or -1."""
    busy = np.flatnonzero(L.acting(L.legal_mask(obs, num_players)))
    return int(busy[0]) if len(busy) else -1


def step_seat(env, obs, seat, action, num_players=2):
    """Step with `action` for `seat`; every other seat sends PASS (ignored)."""
    actions = np.full(num_players, L.PASS, dtype=np.int32)
    actions[seat] = action
    return env.step(actions)


def fresh(num_players=2, seed=7, **kwargs):
    env = Splendor(num_envs=1, num_players=num_players, seed=seed,
                   report_interval=10**9, **kwargs)
    obs, _ = env.reset(seed=seed)
    return env, obs


# --- 1. layout / structural facts -------------------------------------------
def test_layout_constants():
    assert (L.obs_size(2), L.obs_size(3), L.obs_size(4)) == (336, 384, 432)
    for p in (2, 3, 4):
        assert L.num_players_from_obs(L.obs_size(p)) == p
        assert L.turn_offset(p) == L.mask_offset(p) + L.NUM_ACTIONS
        assert L.to_move_offset(p) + 1 == L.obs_size(p)
        assert L.obs_scale(p).shape == (L.obs_size(p),)
        assert (L.obs_scale(p) > 0).all() and L.obs_scale(p).max() == 1.0
    assert L.PLAYERS == 166 and L.CARD_N == 11 and L.PLAYER_N == 48
    assert L.COMBOS3[0] == (0, 1, 2) and L.COMBOS3[-1] == (2, 3, 4)
    assert L.COMBOS2[0] == (0, 1) and L.COMBOS2[-1] == (3, 4)
    assert (L.TAKE3, L.TAKE2, L.TAKE1, L.TAKE2SAME) == (0, 10, 20, 25)
    assert (L.RESERVE_FACEUP, L.RESERVE_DECK) == (30, 42)
    assert (L.BUY_FACEUP, L.BUY_RESERVED, L.PASS) == (45, 57, 60)
    assert (L.DISCARD, L.CHOOSE_NOBLE, L.NUM_ACTIONS) == (61, 67, 72)


def test_random_legal_actions_respects_mask():
    obs = np.zeros((32, L.obs_size(2)), dtype=np.uint8)
    off = L.mask_offset(2)
    obs[:, off + 3] = 1
    obs[:, off + L.PASS] = 1
    rng = np.random.default_rng(0)
    actions = L.random_legal_actions(obs, 2, rng)
    assert set(np.unique(actions)) <= {3, L.PASS}
    assert actions.dtype == np.int32


@needs_ext
def test_card_and_noble_structure():
    num_players = 3
    env = Splendor(num_envs=32, num_players=num_players, report_interval=10**9)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        me = obs[::num_players]
        assert (decks(me) == np.array([36, 26, 16])).all()
        assert (bank(me) == np.array([5] * 5 + [5])).all()
        for slot in range(L.NUM_FACEUP):
            c = card(me, slot)
            cost, bonus, pts = c[:, :5], c[:, 5:10], c[:, 10]
            assert (bonus.sum(axis=1) == 1).all()
            assert cost.max() <= 7
            assert ((cost.sum(axis=1) >= 3) & (cost.sum(axis=1) <= 14)).all()
            tier = slot // 4
            if tier == 0:
                assert (pts <= 1).all()
            elif tier == 1:
                assert ((pts >= 1) & (pts <= 3)).all()
            else:
                assert ((pts >= 3) & (pts <= 5)).all()
        nobles = me[:, L.NOBLES:L.NOBLES + 25].reshape(-1, L.NOBLE_MAX, 5).astype(int)
        sums = nobles.sum(axis=2)
        assert np.isin(sums, (0, 8, 9)).all()
        assert (sums > 0).sum(axis=1).min() == num_players + 1
        assert nobles.max() <= 4
    env.close()


# --- 2. random legal play ---------------------------------------------------
@needs_ext
def test_random_legal_play():
    num_players, num_envs, steps = 2, 256, 4000
    env = Splendor(num_envs=num_envs, num_players=num_players,
                   report_interval=64, seed=3)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    totals = L.initial_tokens(num_players)
    logs, num_terminals, num_discards, num_nobles = [], 0, 0, 0

    for _ in range(steps):
        me = obs[::num_players]
        held = sum(player(me, i)[:, :6] for i in range(num_players))
        assert (bank(me) + held == totals).all(), 'tokens not conserved'

        mask = L.legal_mask(obs, num_players)
        assert mask.any(axis=1).all(), 'every seat needs at least one legal action'
        busy = L.acting(mask).reshape(num_envs, num_players)
        assert busy.sum(axis=1).max() <= 1, 'more than one seat may act'
        to_move = L.to_move(obs, num_players).reshape(num_envs, num_players)
        assert (to_move.sum(axis=1) == 1).all(), 'exactly one seat to move'
        assert (busy <= to_move).all(), 'acting seat must be the seat to move'
        # the 10-token limit: over it only while discarding, never otherwise
        tokens = np.stack([player(obs[s::num_players], 0)[:, :6].sum(axis=1)
                           for s in range(num_players)], axis=1)
        discarding = mask[:, L.DISCARD:L.DISCARD + 6].any(axis=1).reshape(num_envs, num_players)
        assert (tokens[~discarding] <= 10).all(), 'over 10 tokens outside a discard phase'
        assert (tokens[discarding] > 10).all(), 'discard phase at <= 10 tokens'
        num_discards += int(discarding.sum())
        num_nobles += int(mask[:, L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + 5].any(axis=1).sum())

        actions = L.random_legal_actions(obs, num_players, rng)
        obs, rewards, terminals, truncations, info = env.step(actions)
        assert np.abs(rewards).max() < 10
        # terminals are per game: all seats end together
        t = terminals.reshape(num_envs, num_players)
        assert ((t.sum(axis=1) == 0) | (t.sum(axis=1) == num_players)).all()
        num_terminals += int(terminals.sum())
        logs += info

    assert num_terminals > 0, 'no game ever finished'
    assert num_discards > 0, 'random play never had to discard'
    assert logs, 'no logs reported'
    for log in logs:
        for key in ('score', 'points', 'cards', 'nobles', 'game_length',
                    'perf', 'invalid', 'n'):
            assert key in log, f'missing log key {key}'
        assert 0 < log['game_length'] <= env.max_turns
        assert 0 <= log['perf'] <= 1
    env.close()


# --- 3. scripted scenarios --------------------------------------------------
@needs_ext
def test_take3_moves_exactly_three_tokens():
    env, obs = fresh()
    seat = acting_seat(obs, 2)
    assert seat >= 0
    mask = L.legal_mask(obs, 2)[seat]
    idx = int(np.flatnonzero(mask[L.TAKE3:L.TAKE3 + 10])[0])
    combo = L.COMBOS3[idx]

    before_bank, before_me = bank(obs[seat]), player(obs[seat], 0)[:6]
    actions = np.full(2, L.PASS, dtype=np.int32)
    actions[seat] = L.TAKE3 + idx
    obs = env.step(actions)[0]

    after_bank, after_me = bank(obs[seat]), player(obs[seat], 0)[:6]
    for color in range(5):
        delta = 1 if color in combo else 0
        assert after_bank[color] == before_bank[color] - delta
        assert after_me[color] == before_me[color] + delta
    assert after_bank[L.GOLD] == before_bank[L.GOLD]
    env.close()


@needs_ext
def test_take2_same_needs_four_in_bank():
    env, obs = fresh()
    seat = acting_seat(obs, 2)
    assert bank(obs[seat])[0] == 4, '2-player bank starts at 4 per color'
    assert L.legal_mask(obs, 2)[seat][L.TAKE2SAME + 0], 'legal while bank >= 4'

    actions = np.full(2, L.PASS, dtype=np.int32)  # one white token leaves the bank
    actions[seat] = L.TAKE1 + 0
    obs = env.step(actions)[0]

    seat = acting_seat(obs, 2)
    assert bank(obs[seat])[0] == 3
    assert not L.legal_mask(obs, 2)[seat][L.TAKE2SAME + 0], 'illegal while bank < 4'
    assert L.legal_mask(obs, 2)[seat][L.TAKE2SAME + 1], 'other colors still legal'
    env.close()


@needs_ext
def test_reserve_deck_top_gives_gold():
    env, obs = fresh()
    seat = acting_seat(obs, 2)
    assert L.legal_mask(obs, 2)[seat][L.RESERVE_DECK + 0]
    before_bank, before_decks = bank(obs[seat]), decks(obs[seat])
    before_me = player(obs[seat], 0)

    actions = np.full(2, L.PASS, dtype=np.int32)
    actions[seat] = L.RESERVE_DECK + 0
    obs = env.step(actions)[0]

    after_me = player(obs[seat], 0)
    assert after_me[L.GOLD] == before_me[L.GOLD] + 1
    assert bank(obs[seat])[L.GOLD] == before_bank[L.GOLD] - 1
    assert decks(obs[seat])[0] == before_decks[0] - 1
    # the reserved card is present and hidden from the opponent
    assert after_me[L.P_RESERVED + L.CARD_N] == 1
    other = player(obs[1 - seat], 1)
    assert other[L.P_RESERVED + L.CARD_N] == 1
    assert other[L.P_RESERVED:L.P_RESERVED + L.CARD_N].sum() == 0
    env.close()


@needs_ext
def test_discard_phase_and_reserve_gold_at_ten():
    env, obs = fresh(seed=5)
    me = acting_seat(obs, 2)
    other = 1 - me
    # my tokens go 3, 6, 9 (opponent passes via an invalid action, a no-op)
    for _ in range(3):
        idx = int(np.flatnonzero(L.legal_mask(obs, 2)[me][L.TAKE3:L.TAKE3 + 10])[0])
        obs = step_seat(env, obs, me, L.TAKE3 + idx)[0]
        obs = step_seat(env, obs, other, L.PASS)[0]
    assert player(obs[me], 0)[:6].sum() == 9
    # taking 1 -> 10 is fine; taking 2 same on top of that must trigger a discard
    c = int(np.flatnonzero(L.legal_mask(obs, 2)[me][L.TAKE1:L.TAKE1 + 5])[0])
    obs = step_seat(env, obs, me, L.TAKE1 + c)[0]
    obs = step_seat(env, obs, other, L.PASS)[0]
    assert player(obs[me], 0)[:6].sum() == 10
    mask = L.legal_mask(obs, 2)[me]
    assert mask[L.TAKE3:L.TAKE3 + 10].any(), 'takes stay legal at 10 tokens'
    # reserving at 10 tokens still grants the gold (bank has 5) -> 11 -> discard phase
    tier = int(np.flatnonzero(mask[L.RESERVE_DECK:L.RESERVE_DECK + 3])[0])
    turn_before = obs[me, L.turn_offset(2)]
    obs, rewards, terminals, _, _ = step_seat(env, obs, me, L.RESERVE_DECK + tier)
    tokens = player(obs[me], 0)[:6]
    assert tokens.sum() == 11 and tokens[L.GOLD] == 1
    mask = L.legal_mask(obs, 2)[me]
    assert acting_seat(obs, 2) == me, 'the same seat keeps acting'
    assert obs[me, L.turn_offset(2)] == turn_before, 'a sub-step is not a turn'
    assert not mask[:L.DISCARD].any() and mask[L.DISCARD:L.DISCARD + 6].sum() == (tokens > 0).sum()
    assert not L.legal_mask(obs, 2)[other][:L.PASS].any(), 'opponent idle meanwhile'
    # an illegal action inside the phase is forced to a legal discard, never a stall
    obs = step_seat(env, obs, me, L.TAKE3)[0]
    assert player(obs[me], 0)[:6].sum() == 10
    assert acting_seat(obs, 2) == other, 'turn passes once back at 10'
    assert obs[other, L.turn_offset(2)] == turn_before + 1
    # explicit discard: take 3 -> 13, return gold, then two colors
    obs = step_seat(env, obs, other, L.PASS)[0]
    idx = int(np.flatnonzero(L.legal_mask(obs, 2)[me][L.TAKE3:L.TAKE3 + 10])[0])
    obs = step_seat(env, obs, me, L.TAKE3 + idx)[0]
    assert player(obs[me], 0)[:6].sum() == 13
    bank_gold = bank(obs[me])[L.GOLD]
    obs = step_seat(env, obs, me, L.DISCARD + L.GOLD)[0]
    assert player(obs[me], 0)[L.GOLD] == 0 and bank(obs[me])[L.GOLD] == bank_gold + 1
    for _ in range(2):
        c = int(np.flatnonzero(L.legal_mask(obs, 2)[me][L.DISCARD:L.DISCARD + 5])[0])
        obs = step_seat(env, obs, me, L.DISCARD + c)[0]
    assert player(obs[me], 0)[:6].sum() == 10 and acting_seat(obs, 2) == other
    env.close()


@needs_ext
def test_noble_choice_when_several_qualify():
    env, obs = fresh(seed=3)
    me = acting_seat(obs, 2)
    other = 1 - me
    bonuses = [0] * 10
    bonuses[me * 5:me * 5 + 5] = [7] * 5          # everything affordable, all nobles met
    env.put_state(0, bonuses=bonuses)
    obs = env.observations
    mask = L.legal_mask(obs, 2)[me]
    assert mask[L.BUY_FACEUP:L.BUY_FACEUP + 12].all(), 'bonuses cover every cost'
    assert not mask[L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + 5].any(), 'nobles only visit after a buy'
    points_before = player(obs[me], 0)[L.P_POINTS]
    card_points = card(obs[me], 0)[10]
    obs, rewards, _, _, _ = step_seat(env, obs, me, L.BUY_FACEUP)
    mask = L.legal_mask(obs, 2)[me]
    assert acting_seat(obs, 2) == me
    assert not mask[:L.CHOOSE_NOBLE].any(), 'only noble choices are legal'
    choices = np.flatnonzero(mask[L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + 5])
    assert len(choices) == 3, 'all num_players + 1 dealt nobles qualify'
    assert player(obs[me], 0)[L.P_POINTS] == points_before + card_points, 'no noble yet'
    pick = int(choices[1])
    noble_req = obs[me, L.NOBLES + 5 * pick:L.NOBLES + 5 * pick + 5].copy()
    assert noble_req.sum() > 0
    obs, rewards, _, _, _ = step_seat(env, obs, me, L.CHOOSE_NOBLE + pick)
    assert player(obs[me], 0)[L.P_POINTS] == points_before + card_points + 3
    assert rewards[me] == pytest.approx(0.02 * 3)
    assert obs[me, L.NOBLES + 5 * pick:L.NOBLES + 5 * pick + 5].sum() == 0, 'noble slot emptied'
    assert (obs[me, L.NOBLES:L.NOBLES + 25].reshape(5, 5).sum(axis=1) > 0).sum() == 2
    assert acting_seat(obs, 2) == other, 'turn ends after the choice'
    # one more buy: two nobles still qualify -> another choice; exactly one -> automatic
    obs = step_seat(env, obs, other, L.PASS)[0]
    obs = step_seat(env, obs, me, L.BUY_FACEUP + 1)[0]
    assert len(np.flatnonzero(L.legal_mask(obs, 2)[me][L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + 5])) == 2
    obs = step_seat(env, obs, me, L.CHOOSE_NOBLE + int(np.flatnonzero(
        L.legal_mask(obs, 2)[me][L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + 5])[0]))[0]
    obs = step_seat(env, obs, other, L.PASS)[0]
    pts = player(obs[me], 0)[L.P_POINTS]
    obs = step_seat(env, obs, me, L.BUY_FACEUP + 2)[0]
    assert acting_seat(obs, 2) == other, 'a single qualifying noble is taken automatically'
    assert player(obs[me], 0)[L.P_POINTS] >= pts + 3
    env.close()


@needs_ext
def test_state_snapshot_roundtrip():
    env, obs = fresh(seed=9)
    before = obs.copy()
    snap = env.get_state(0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        obs = env.step(L.random_legal_actions(obs, 2, rng))[0]
    assert (obs != before).any()
    env.put_state(0, state=snap)
    assert (env.observations == before).all(), 'snapshot restore must reproduce the obs'
    env.close()


@needs_ext
def test_idle_seats_are_pass_only():
    env, obs = fresh(num_players=3)
    seat = acting_seat(obs, 3)
    mask = L.legal_mask(obs, 3)
    for other in range(3):
        if other == seat:
            continue
        assert mask[other][L.PASS]
        assert not mask[other][:L.PASS].any()
    env.close()


@needs_ext
def test_invalid_action_is_a_noop():
    env, obs = fresh()
    seat = acting_seat(obs, 2)
    mask = L.legal_mask(obs, 2)[seat]
    illegal = int(np.flatnonzero(~mask[L.BUY_FACEUP:L.BUY_FACEUP + 12])[0])
    before = obs[:, :L.mask_offset(2)].copy()
    before_turn = obs[seat, L.turn_offset(2)]

    actions = np.full(2, L.PASS, dtype=np.int32)
    actions[seat] = L.BUY_FACEUP + illegal
    obs, rewards, terminals, _, _ = env.step(actions)

    assert (obs[:, :L.mask_offset(2)] == before).all(), 'state changed'
    assert obs[seat, L.turn_offset(2)] == before_turn + 1
    assert rewards[seat] == 0 and not terminals.any()
    env.close()


def _take_toward(obs_row, slot, mask):
    """Pick a take action that moves us closer to affording `slot`."""
    cost = card(obs_row, slot)[:5]
    me = player(obs_row, 0)
    need = np.maximum(cost - me[L.P_BONUS:L.P_BONUS + 5] - me[:5], 0)
    stock = bank(obs_row)[:5]
    want = [c for c in range(5) if need[c] > 0 and stock[c] > 0]
    for i, combo in enumerate(L.COMBOS3):
        if mask[L.TAKE3 + i] and all(c in want for c in combo):
            return L.TAKE3 + i
    for i, combo in enumerate(L.COMBOS2):
        if mask[L.TAKE2 + i] and all(c in want for c in combo):
            return L.TAKE2 + i
    for c in want:
        if need[c] >= 2 and mask[L.TAKE2SAME + c]:
            return L.TAKE2SAME + c
    for c in want:
        if mask[L.TAKE1 + c]:
            return L.TAKE1 + c
    return L.PASS


@needs_ext
def test_buy_card_after_collecting_tokens():
    env, obs = fresh(seed=11)
    target = acting_seat(obs, 2)
    slot = min(range(4), key=lambda s: card(obs[target], s)[:5].sum())
    bought = False

    for _ in range(60):
        actions = np.full(2, L.PASS, dtype=np.int32)
        seat = acting_seat(obs, 2)
        if seat != target:
            obs = env.step(actions)[0]
            continue

        mask = L.legal_mask(obs, 2)[target]
        if mask[L.BUY_FACEUP + slot]:
            before_card = card(obs[target], slot)
            before_me = player(obs[target], 0)
            before_bank = bank(obs[target])
            actions[target] = L.BUY_FACEUP + slot
            obs = env.step(actions)[0]

            after_me = player(obs[target], 0)
            color = int(np.argmax(before_card[5:10]))
            assert after_me[L.P_BONUS + color] == before_me[L.P_BONUS + color] + 1
            assert after_me[L.P_POINTS] >= before_me[L.P_POINTS] + before_card[10]
            assert after_me[:6].sum() < before_me[:6].sum(), 'no tokens paid'
            assert bank(obs[target]).sum() > before_bank.sum(), 'tokens not returned'
            refill = card(obs[target], slot)
            assert (refill != before_card).any(), 'slot was not refilled'
            assert refill[5:10].sum() == 1, 'refilled slot must hold a card'
            bought = True
            break

        actions[target] = _take_toward(obs[target], slot, mask)
        obs = env.step(actions)[0]

    assert bought, 'never managed to buy a tier-1 card'
    env.close()
