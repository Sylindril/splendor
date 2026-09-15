#!/usr/bin/env python3
"""Zero-dependency local web GUI for the Splendor PufferLib environment.

    python gui.py [--load-model-path experiments/x.pt] [--num-players 2]
                  [--human 0|none] [--port 8000] [--mcts SIMS] [--sample]

Serves a single self-contained page (splendor/gui/index.html) plus a tiny JSON
API.  Every non-human seat is a `splendor.agents.Agent` built from a string
spec -- 'random', 'greedy', a checkpoint path, 'mcts:PATH:SIMS' or
'puffer5:PATH:HIDDEN:LAYERS' (PufferLib 5.0 weights) -- and the
page can re-deal the table with a different line-up through POST /config.
"""
import argparse
import glob
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from splendor import layout as L

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, 'splendor', 'gui', 'index.html')
EXPERIMENTS = os.path.join(HERE, 'experiments')
MODELS = os.path.join(HERE, 'models')  # shipped checkpoints (tracked in git)
COLORS = L.COLOR_NAMES
BOTS = ('random', 'greedy')
MCTS_SIMS = 64                 # default simulations for a bare 'mcts:PATH'
EVENT_LOG = 24                 # moves kept for the page's animation queue


def action_text(a):
    """One-line English description of an action index."""
    if a is None:
        return ''
    a = int(a)
    if a < L.TAKE2:
        return 'take ' + ', '.join(COLORS[c] for c in L.COMBOS3[a - L.TAKE3])
    if a < L.TAKE1:
        return 'take ' + ', '.join(COLORS[c] for c in L.COMBOS2[a - L.TAKE2])
    if a < L.TAKE2SAME:
        return 'take 1 %s' % COLORS[a - L.TAKE1]
    if a < L.RESERVE_FACEUP:
        return 'take 2 %s' % COLORS[a - L.TAKE2SAME]
    if a < L.RESERVE_DECK:
        s = a - L.RESERVE_FACEUP
        return 'reserve tier %d card %d' % (s // 4 + 1, s % 4 + 1)
    if a < L.BUY_FACEUP:
        return 'reserve tier %d deck top' % (a - L.RESERVE_DECK + 1)
    if a < L.BUY_RESERVED:
        s = a - L.BUY_FACEUP
        return 'buy tier %d card %d' % (s // 4 + 1, s % 4 + 1)
    if a < L.PASS:
        return 'buy reserved card %d' % (a - L.BUY_RESERVED + 1)
    if a == L.PASS:
        return 'pass'
    if a < L.CHOOSE_NOBLE:
        return 'return 1 %s' % COLORS[a - L.DISCARD]
    return 'choose noble %d' % (a - L.CHOOSE_NOBLE + 1)


def card(obs, off):
    """Decode CARD_N bytes; None for an empty slot."""
    blk = obs[off:off + L.CARD_N]
    if not blk.any():
        return None
    bonus = blk[5:10]
    return {'cost': [int(x) for x in blk[:5]],
            'bonus': int(np.argmax(bonus)) if bonus.any() else -1,
            'points': int(blk[10])}


def paid_tokens(cost, bonuses, tokens):
    """Tokens a buyer hands back to the bank: bonuses first, then gems, then gold.

    Mirrors the engine's payment order, so it is exact and -- unlike diffing the
    pre/post token vectors -- it still works on the step that ends the game."""
    pay, short = [0] * 6, 0
    for i in range(L.NUM_COLORS):
        need = max(0, int(cost[i]) - int(bonuses[i]))
        pay[i] = min(need, int(tokens[i]))
        short += need - pay[i]
    pay[L.GOLD] = short
    return pay


def visible(c):
    """A card dict, or None when the slot is empty or hidden from the viewer."""
    return None if (c is None or c.get('hidden')) else c


# ============================ agent specs ============================
# A seat agent is described by a string:
#   'random' / 'greedy'      the built-in yardsticks
#   'experiments/latest.pt'  a policy checkpoint (greedy, or sampled if --sample)
#   'mcts:PATH:SIMS'         PUCT search over that checkpoint (SIMS default 64)
#   'puffer5:PATH:H:L'       a PufferLib 5.0 '*_weights.bin' export (H x L MinGRU)

PUFFER5 = 'puffer5:'


def is_puffer5(spec):
    return isinstance(spec, str) and spec.strip().startswith(PUFFER5)

_scan = {}          # path -> ((mtime, size), num_players or None)


def checkpoint_players(path):
    """Player count a checkpoint was trained for, from its encoder width."""
    import torch
    try:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:        # half-written by a running trainer, or not a policy
        return None
    while isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    if not isinstance(sd, dict):
        return None
    w = sd.get('encoder.0.weight', sd.get('module.encoder.0.weight'))
    if w is None or getattr(w, 'ndim', 0) != 2:
        return None
    width = int(w.shape[1])
    for p in (2, 3, 4):
        if L.obs_size(p) == width:
            return p
    return None


def checkpoints(roots=(MODELS, EXPERIMENTS)):
    """[{path, label, num_players, mtime}] for every usable .pt under models/
    and experiments/.

    torch.load only re-runs for files whose (mtime, size) changed, so a training
    run dropping new checkpoints is picked up without re-reading the rest."""
    out = []
    paths = [(root, p) for root in roots
             for p in glob.glob(os.path.join(root, '**', '*.pt'), recursive=True)]
    for root, path in paths:
        if os.path.basename(path) == 'trainer_state.pt':
            continue
        try:
            st = os.stat(path)
        except OSError:
            continue
        key = (st.st_mtime, st.st_size)
        hit = _scan.get(path)
        if hit is None or hit[0] != key:
            hit = (key, checkpoint_players(path))
            _scan[path] = hit
        if hit[1] is None:
            continue
        out.append({'path': os.path.relpath(path, HERE), 'kind': 'pt',
                    'label': os.path.relpath(path, root if root == EXPERIMENTS else HERE),
                    'num_players': hit[1], 'mtime': st.st_mtime})
    out += puffer5_checkpoints(roots)
    out.sort(key=lambda c: -c['mtime'])
    return out


def puffer5_checkpoints(roots=(MODELS, EXPERIMENTS)):
    """The same, for PufferLib 5.0 '*_weights.bin' exports.

    Those files are a flat float32 blob, so the architecture is solved for from
    the file size; 'path' is the full agent spec ('puffer5:REL:HIDDEN:LAYERS')
    because that is what the page hands back as the seat's spec."""
    from splendor import puffernet as PN
    out = []
    for root in roots:
        for path in glob.glob(os.path.join(root, '**', '*_weights.bin'),
                              recursive=True):
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = (st.st_mtime, st.st_size)
            hit = _scan.get(path)
            if hit is None or hit[0] != key:
                hit = (key, PN.infer_arch(path))
                _scan[path] = hit
            if hit[1] is None:            # not a PufferNet weight file
                continue
            num_players, hidden, layers = hit[1]
            spec = PN.spec_for(os.path.relpath(path, HERE), hidden, layers)
            out.append({'path': spec, 'kind': 'puffer5',
                        'label': PN.spec_label(spec),
                        'num_players': num_players, 'mtime': st.st_mtime})
    return out


def parse_spec(spec):
    """'mcts:PATH:SIMS' -> (PATH, sims); every other spec -> (spec, None)."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError('empty agent spec')
    spec = spec.strip()
    if not spec.startswith('mcts:'):
        return spec, None
    rest, sims = spec[len('mcts:'):], MCTS_SIMS
    if ':' in rest:
        head, tail = rest.rsplit(':', 1)
        if tail.strip().isdigit():
            rest, sims = head, int(tail)
    if not rest:
        raise ValueError('mcts needs a checkpoint: mcts:PATH:SIMS')
    return rest, max(1, min(sims, 100000))


def resolve(path, num_players=2):
    """A checkpoint path as given, relative to the project directory, or
    'latest' = the newest checkpoint trained for `num_players`."""
    if path == 'latest':
        from splendor.agents import latest_checkpoint
        return latest_checkpoint(num_players)
    for p in (path, os.path.join(HERE, path)):
        if os.path.isfile(p):
            return p
    raise ValueError('no such checkpoint: %s' % path)


def spec_label(spec):
    """Human-readable name for a spec, e.g. 'latest.pt . MCTS 64'."""
    if is_puffer5(spec):
        from splendor import puffernet as PN
        return PN.spec_label(spec)
    base, sims = parse_spec(spec)
    if base in BOTS:
        return base
    try:
        name = os.path.relpath(resolve(base), EXPERIMENTS)
    except ValueError:
        name = base
    if name.startswith('..'):
        name = os.path.basename(base)
    return name if sims is None else '%s · MCTS %d' % (name, sims)


def build_agent(spec, num_players, sample=False, seed=0, max_turns=None):
    """Agent for one seat. Raises ValueError with a message the page can show."""
    from splendor import agents
    base, sims = parse_spec(spec)
    temperature = 1.0 if sample else 0.0
    if is_puffer5(base):
        from splendor import puffernet as PN
        if sims is not None:
            raise ValueError('search needs a .pt checkpoint, not PufferLib 5.0 weights')
        weights, hidden, layers = PN.parse_spec(base)
        arch = PN.infer_arch(resolve(weights))
        if arch is None:
            raise ValueError('%s is not a PufferNet weight file'
                             % os.path.basename(weights))
        if arch[0] != num_players:
            raise ValueError('%s is a %d-player weight file, this table has %d seats'
                             % (os.path.basename(weights), arch[0], num_players))
        return agents.make_agent(PN.spec_for(resolve(weights), hidden, layers),
                                 num_players, seed=seed, temperature=temperature,
                                 name=spec_label(base))
    if base in BOTS:
        if sims is not None:
            raise ValueError('search needs a checkpoint, not %r' % base)
        return agents.make_agent(base, num_players, seed=seed)
    path = resolve(base, num_players)
    P = checkpoint_players(path)
    if P is None:
        raise ValueError('%s is not a readable policy checkpoint' % base)
    if P != num_players:
        raise ValueError('%s is a %d-player checkpoint, this table has %d seats'
                         % (os.path.basename(base), P, num_players))
    if sims is None:
        return agents.make_agent(path, num_players, seed=seed,
                                 temperature=temperature, name=spec_label(spec))
    from splendor.mcts import MCTSAgent
    return MCTSAgent(path, num_players=num_players, sims=sims, seed=seed,
                     temperature=temperature, capacity=1, max_turns=max_turns,
                     name=spec_label(spec))


def close_agent(a):
    if a is not None and hasattr(a, 'close'):
        try:
            a.close()
        except Exception:
            pass


class Game:
    """Owns the env (num_envs=1) and every mutation of it, under one lock."""

    def __init__(self, args):
        self.lock = threading.RLock()
        self.seed = args.seed
        self.max_turns = args.max_turns
        self.reward_point = 0.02
        self.rng = random.Random(args.seed)
        self.env = None
        self.seat_agents, self.specs = [], []
        self.P, self.human, self.sample = args.num_players, args.human, args.sample
        self.last, self.over, self.cur = '', None, 0
        self.game_over = False
        self.event, self.event_id, self.event_log = None, 0, []
        self.last_action, self.last_seat = None, None
        self.bought, self.nobles_won = [], []
        self.configure(args.num_players, args.human, args.seats, args.sample)

    # ------------------------------------------------------------- line-up
    def _new_env(self, num_players):
        from splendor import Splendor
        # auto_reset=False: the env HOLDS the finished board (all-zero legal mask,
        # further steps are no-ops) until reset(), so the page can show and let
        # the user inspect the real final position instead of a fresh deal.
        return Splendor(num_envs=1, num_players=num_players,
                        max_turns=self.max_turns, reward_point=self.reward_point,
                        reward_card=0.0, reward_win=1.0, reward_loss=-1.0,
                        auto_reset=False, seed=self.seed)

    def configure(self, num_players, human, seats, sample):
        """Rebuild env + seat agents and deal a new game. ValueError if invalid.

        Everything is built before anything is swapped, so a rejected config
        leaves the running game untouched."""
        P = int(num_players)
        if not 2 <= P <= 4:
            raise ValueError('players must be 2, 3 or 4')
        if human is not None:
            human = int(human)
            if not 0 <= human < P:
                raise ValueError('your seat must be in [0, %d) or "watch"' % P)
        seats = list(seats if seats is not None else [None] * P)
        if len(seats) != P:
            raise ValueError('need one agent per seat (%d given, %d seats)'
                             % (len(seats), P))
        specs, built = [], []
        try:
            for s in range(P):
                if s == human:
                    specs.append(None)
                    built.append(None)
                    continue
                if not seats[s]:
                    raise ValueError('seat %d needs an agent' % s)
                specs.append(str(seats[s]).strip())
                built.append(build_agent(specs[s], P, sample=bool(sample),
                                         seed=self.seed + s,
                                         max_turns=self.max_turns))
        except Exception:
            for a in built:
                close_agent(a)
            raise
        with self.lock:
            for a in self.seat_agents:
                close_agent(a)
            old, self.env = self.env, self._new_env(P)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            self.P, self.human, self.sample = P, human, bool(sample)
            self.seat_agents, self.specs = built, specs
            self.new_game()

    def agent_names(self):
        return [None if a is None else a.name for a in self.seat_agents]

    # ------------------------------------------------------------------ env
    def mask(self, seat):
        off = L.mask_offset(self.P)
        return np.asarray(self.env.observations[seat][off:off + L.NUM_ACTIONS]) > 0

    def _rescan(self, advanced=False):
        # A seat is "acting" if it has any legal action other than PASS.  During a
        # discard / noble sub-phase only actions >= DISCARD are legal, so testing
        # mask[:PASS] alone would miss it -- use layout.acting().
        live = [s for s in range(self.P) if L.to_move(self.env.observations[s], self.P)[0]]
        self.cur = live[0] if live else (self.cur + 1) % self.P if advanced else self.cur

    def phase(self, seat):
        """'act', 'discard' or 'noble' for `seat`, from its legal mask."""
        m = self.mask(seat)
        if m[L.DISCARD:L.DISCARD + 6].any():
            return 'discard'
        if m[L.CHOOSE_NOBLE:L.CHOOSE_NOBLE + 5].any():
            return 'noble'
        return 'act'

    def _reset_agents(self):
        for a in self.seat_agents:
            if a is not None:
                a.reset()

    def new_game(self):
        with self.lock:
            self.env.reset()
            self._reset_agents()
            self.last, self.over, self.game_over = '', None, False
            self.event, self.last_action, self.last_seat = None, None, None
            self.event_id, self.event_log = 0, []
            self.bought = [[] for _ in range(self.P)]
            self.nobles_won = [[] for _ in range(self.P)]
            self.cur = 0
            self._rescan()

    def act(self, seat, action):
        """Apply one action for `seat`; returns True if the game just ended."""
        if self.game_over:
            return True
        mask = self.mask(seat)
        if not mask[action]:
            action = L.PASS
        # `pre` is the mover's own view (it sees its own deck reserves), `pre_v`
        # the view the page is rendered from -- a deck reserve stays hidden there.
        pre = self.decode(seat)
        viewer = self.human if self.human is not None else seat
        pre_v = pre if viewer == seat else self.decode(viewer)
        self.env.actions[:] = L.PASS
        self.env.actions[seat] = action
        self.env.step(self.env.actions)
        self.last = 'seat %d: %s' % (seat, action_text(action))
        done = bool(np.any(self.env.terminals[:self.P]))
        # The env holds the finished board (auto_reset=False), so the post-state
        # still belongs to THIS game even on the step that ended it.
        post = self.decode(seat)
        post_v = post if viewer == seat else self.decode(viewer)
        self._record(seat, int(action), pre, pre_v, post, post_v)
        if done:
            self.game_over = True
            # `bought` / `nobles_won` are NOT cleared: the final mats stay
            # inspectable until the user asks for a new game.
            self.over = self._outcome(post)
            self._rescan()
        else:
            self._rescan(advanced=True)
        return done

    # --------------------------------------------------------------- events
    def _record(self, seat, action, pre, pre_v, post, post_v):
        """Build self.event for the move just made and update self.bought.

        `pre` / `post` are the mover's own view, `pre_v` / `post_v` the viewer's;
        `post*` are None on the step that ended the game."""
        self.last_action, self.last_seat = action, seat
        self.event_id += 1
        self.event = self._event(seat, action, pre, pre_v, post, post_v)
        # One POST can contain several moves (a human action followed by every
        # agent up to the human's next turn), so keep a short rolling log: the
        # page animates every entry newer than the one it last played.
        self.event_log.append(dict(self.event, id=self.event_id,
                                   action=action, text=action_text(action)))
        del self.event_log[:-EVENT_LOG]

    def _taken_noble(self, pre_v, post_v):
        """{slot, req} for a noble that left the table this step, else None."""
        if post_v is None:
            return None
        for i, req in enumerate(pre_v['nobles']):
            if req is not None and post_v['nobles'][i] is None:
                return {'slot': i, 'req': list(req)}
        return None

    def _event(self, seat, a, pre, pre_v, post, post_v):
        mine = pre['players'][seat]                 # the mover, in its own view
        if a < L.TAKE2:
            return {'type': 'take', 'seat': seat, 'gems': list(L.COMBOS3[a - L.TAKE3])}
        if a < L.TAKE1:
            return {'type': 'take', 'seat': seat, 'gems': list(L.COMBOS2[a - L.TAKE2])}
        if a < L.TAKE2SAME:
            return {'type': 'take', 'seat': seat, 'gems': [a - L.TAKE1]}
        if a < L.RESERVE_FACEUP:
            c = a - L.TAKE2SAME
            return {'type': 'take2', 'seat': seat, 'gems': [c, c]}
        if a < L.BUY_FACEUP:                        # reserve, face-up or deck top
            deck = a >= L.RESERVE_DECK
            src = ({'kind': 'deck', 'tier': a - L.RESERVE_DECK} if deck else
                   {'kind': 'faceup', 'slot': a - L.RESERVE_FACEUP})
            to_slot = self._new_slot(mine['reserved'],
                                     post and post['players'][seat]['reserved'])
            # A deck reserve is only knowable from the post-state, and only to a
            # viewer who owns it: everyone else sees a face-down back.
            if deck:
                card = None
                if post is not None and self.human in (None, seat):
                    card = visible(post['players'][seat]['reserved'][to_slot])
            else:
                card = visible(pre_v['faceup'][src['slot']])
            gold = 0
            if post is not None:
                gold = 1 if post['players'][seat]['tokens'][L.GOLD] > \
                            mine['tokens'][L.GOLD] else 0
            elif pre['bank'][L.GOLD] > 0:
                gold = 1
            return {'type': 'reserve', 'seat': seat, 'from': src, 'card': card,
                    'to_slot': to_slot, 'gold': gold}
        if a < L.BUY_RESERVED:                      # buy a face-up slot
            slot = a - L.BUY_FACEUP
            return self._buy(seat, {'kind': 'faceup', 'slot': slot},
                             pre['faceup'][slot], visible(pre_v['faceup'][slot]),
                             mine, pre_v, post_v)
        if a < L.PASS:                              # buy one of my reserved cards
            idx = a - L.BUY_RESERVED
            own = mine['reserved'][idx]             # never hidden in my own view
            shown = visible(pre_v['players'][seat]['reserved'][idx])
            return self._buy(seat, {'kind': 'reserved', 'idx': idx},
                             own, shown, mine, pre_v, post_v)
        if a == L.PASS:
            return {'type': 'pass', 'seat': seat}
        if a < L.CHOOSE_NOBLE:
            return {'type': 'discard', 'seat': seat, 'color': a - L.DISCARD}
        slot = a - L.CHOOSE_NOBLE
        req = pre_v['nobles'][slot] or [0] * 5
        self.nobles_won[seat].append(list(req))
        return {'type': 'noble', 'seat': seat, 'slot': slot, 'req': list(req)}

    def _buy(self, seat, src, card, shown, mine, pre_v, post_v):
        """Buy event; `card` is the mover's own (always visible) copy."""
        paid = [0] * 6
        if card is not None:
            self.bought[seat].append(card)
            paid = paid_tokens(card['cost'], mine['bonuses'], mine['tokens'])
        noble = self._taken_noble(pre_v, post_v)
        if noble is not None:
            self.nobles_won[seat].append(list(noble['req']))
        return {'type': 'buy', 'seat': seat, 'from': src, 'card': shown,
                'paid': paid, 'noble': noble}

    @staticmethod
    def _new_slot(pre_res, post_res):
        """Reserved slot a card just landed in (first free slot as a fallback)."""
        if post_res is not None:
            for i in range(3):
                if pre_res[i] is None and post_res[i] is not None:
                    return i
        for i in range(3):
            if pre_res[i] is None:
                return i
        return 0

    def _winners(self):
        """(winners, draw) from the terminal rewards.

        +1 win / -1 loss / 0 draw for every seat, plus at most 0.02 * points of
        shaping for the mover -- far inside the 0.5 thresholds below."""
        r = [float(x) for x in np.asarray(self.env.rewards[:self.P])]
        if max(r) > 0.5:
            return [s for s in range(self.P) if r[s] > 0.5], False
        winners = [s for s in range(self.P) if r[s] > -0.5]
        return winners, len(winners) != 1

    def _outcome(self, final):
        """The result panel's payload, read off the real FINAL board."""
        winners, draw = self._winners()
        pts = [p['points'] for p in final['players']]
        return {
            'winners': winners, 'draw': draw, 'points': pts,
            'cards': [int(sum(p['bonuses'])) for p in final['players']],
            'nobles': [len(self.nobles_won[s]) for s in range(self.P)],
            'turns': int(final['turn']),
            'ended_by': 'points' if max(pts) >= 15 else 'turn_limit',
        }

    def agent_turn(self, limit=400):
        """Play agent seats until it is the human's turn (or a game ends)."""
        if self.human is None:
            return
        with self.lock:
            if self.game_over:
                return
            for _ in range(limit):
                seat = self.cur
                if self.human is not None and seat == self.human:
                    return
                if self.step_one():
                    return

    def step_one(self):
        """One agent move for the acting seat. True if the game ended."""
        with self.lock:
            if self.game_over:
                return True
            seat = self.cur
            mask = self.mask(seat)
            legal = np.flatnonzero(mask)
            agent = self.seat_agents[seat]
            if agent is None or len(legal) == 0:      # human seat / nothing legal
                action = int(self.rng.choice(list(legal))) if len(legal) else L.PASS
            else:
                # num_envs == 1, so the agent sees exactly one row: this seat's.
                obs = np.asarray(self.env.observations[seat:seat + 1])
                a = agent.act(obs, mask.reshape(1, -1), self.env, np.array([0]))
                action = int(np.asarray(a).reshape(-1)[0])
            return self.act(seat, action)

    # --------------------------------------------------------------- decode
    def decode(self, view):
        """Full board state as seen from seat `view` (perspective-relative obs)."""
        obs = np.asarray(self.env.observations[view])
        P = self.P
        # Fixed NOBLE_MAX-long list with None holes: CHOOSE_NOBLE indexes the
        # slot, not the n-th remaining noble.
        nobles = []
        for i in range(L.NOBLE_MAX):
            o = L.NOBLES + 5 * i
            req = [int(x) for x in obs[o:o + 5]]
            nobles.append(req if any(req) else None)
        players = [None] * P
        for i in range(P):
            b = L.player_offset(i)
            # 3 fixed slots (holes kept: BUY_RESERVED indexes the slot, not the
            # n-th card).  None = empty, {'hidden': True} = opponent's deck reserve.
            res = []
            for r in range(3):
                c = b + L.P_RESERVED + L.RESERVED_N * r
                present = bool(obs[c + L.CARD_N])
                res.append((card(obs, c) or {'hidden': True}) if present else None)
            s = (view + i) % P
            players[s] = {
                'seat': s,
                'tokens': [int(x) for x in obs[b:b + 6]],
                'bonuses': [int(x) for x in obs[b + L.P_BONUS:b + L.P_BONUS + 5]],
                'points': int(obs[b + L.P_POINTS]),
                'reserved': res,
                # cards this seat has bought this game, and the nobles it won:
                # neither is in the observation, both are tracked in act().
                'bought': list(self.bought[s]) if s < len(self.bought) else [],
                'nobles': [list(n) for n in self.nobles_won[s]]
                          if s < len(self.nobles_won) else [],
            }
        return {
            'bank': [int(x) for x in obs[L.BANK:L.BANK + 6]],
            'decks': [int(x) for x in obs[L.DECKS:L.DECKS + 3]],
            'faceup': [card(obs, L.FACEUP + L.CARD_N * s) for s in range(12)],
            'nobles': nobles,
            'players': players,
            'turn': int(obs[L.turn_offset(P)]),
        }

    def state(self):
        with self.lock:
            view = self.human if self.human is not None else self.cur
            st = self.decode(view)
            has_model = any(s and parse_spec(s)[0] not in BOTS for s in self.specs)
            st.update(
                num_players=self.P, acting=self.cur, view=view, human=self.human,
                mask=[int(x) for x in self.mask(view)],
                acting_mask=[int(x) for x in self.mask(self.cur)],
                last=self.last, over=self.over, game_over=self.game_over,
                has_model=has_model,
                event=self.event, event_id=self.event_id,
                events=list(self.event_log),
                last_action=self.last_action, last_seat=self.last_seat,
                seats=list(self.specs), agent_names=self.agent_names(),
                sample=self.sample,
                phase=self.phase(self.cur),
                actions={'TAKE3': L.TAKE3, 'TAKE2': L.TAKE2, 'TAKE1': L.TAKE1,
                         'TAKE2SAME': L.TAKE2SAME, 'RESERVE_FACEUP': L.RESERVE_FACEUP,
                         'RESERVE_DECK': L.RESERVE_DECK, 'BUY_FACEUP': L.BUY_FACEUP,
                         'BUY_RESERVED': L.BUY_RESERVED, 'PASS': L.PASS,
                         'DISCARD': L.DISCARD, 'CHOOSE_NOBLE': L.CHOOSE_NOBLE},
                combos3=[list(c) for c in L.COMBOS3],
                combos2=[list(c) for c in L.COMBOS2],
                colors=list(COLORS),
            )
            return st

    def options(self):
        """Everything the in-page table picker needs."""
        found = checkpoints()                       # slow-ish: scan outside the lock
        with self.lock:
            current = {'num_players': self.P, 'human': self.human,
                       'seats': list(self.specs), 'sample': self.sample}
        return {'checkpoints': found, 'bots': list(BOTS), 'current': current}


class Handler(BaseHTTPRequestHandler):
    game = None

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split('?')[0] in ('/', '/index.html'):
            with open(INDEX, 'rb') as f:
                return self._send(200, f.read(), 'text/html; charset=utf-8')
        if self.path.startswith('/state'):
            return self._send(200, json.dumps(self.game.state()))
        if self.path.startswith('/options'):
            return self._send(200, json.dumps(self.game.options()))
        self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(n) or b'{}')
        except ValueError:
            body = {}
        g, path = self.game, self.path.split('?')[0]
        try:
            if path == '/reset':
                g.new_game()
                if g.human is not None:
                    g.agent_turn()
            elif path == '/config':
                try:
                    g.configure(body.get('num_players', g.P),
                                body.get('human', g.human),
                                body.get('seats'),
                                body.get('sample', g.sample))
                except Exception as e:
                    msg = str(e) or repr(e)
                    return self._send(400, json.dumps({'error': msg}))
                if g.human is not None:
                    g.agent_turn()
            elif path == '/step':
                with g.lock:
                    if g.game_over:          # a finished board: stepping is a no-op
                        pass
                    elif g.human is None:
                        g.step_one()
                    else:
                        g.agent_turn()
            elif path == '/act':
                with g.lock:
                    if g.game_over:
                        return self._send(400, json.dumps({'error': 'game over'}))
                    if g.human is None:
                        return self._send(400, json.dumps({'error': 'watch mode'}))
                    a = int(body.get('action', L.PASS))
                    if not 0 <= a < L.NUM_ACTIONS or g.cur != g.human \
                            or not g.mask(g.human)[a]:
                        return self._send(400, json.dumps({'error': 'illegal action'}))
                    if not g.act(g.human, a):
                        g.agent_turn()
            else:
                return self._send(404, json.dumps({'error': 'not found'}))
        except Exception as e:  # never kill the server on a bad request
            return self._send(500, json.dumps({'error': repr(e)}))
        self._send(200, json.dumps(g.state()))


def main():
    ap = argparse.ArgumentParser(description='Splendor web GUI')
    ap.add_argument('--load-model-path', default=None)
    ap.add_argument('--num-players', type=int, default=2)
    ap.add_argument('--max-turns', type=int, default=None,
                    help='total turns before a draw (env default: 60 per seat)')
    ap.add_argument('--human', default='0', help="seat index, or 'none' to watch")
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mcts', type=int, default=None, metavar='SIMS',
                    help='agent seats search with MCTS over --load-model-path')
    ap.add_argument('--rnn', action='store_true',
                    help='(removed) recurrent checkpoints are no longer supported')
    ap.add_argument('--sample', action='store_true', help='sample instead of argmax')
    args = ap.parse_args()
    args.human = None if str(args.human).lower() in ('none', '-1', '') else int(args.human)
    if args.human is not None and not 0 <= args.human < args.num_players:
        ap.error('--human must be a seat in [0, num_players)')
    if args.rnn:
        print('warning: --rnn is ignored (the seat-agent API has no LSTM path)',
              flush=True)

    spec = 'random'
    if args.load_model_path:
        spec = ('mcts:%s:%d' % (args.load_model_path, args.mcts) if args.mcts
                else args.load_model_path)
    elif args.mcts:
        ap.error('--mcts needs --load-model-path')
    args.seats = [None if s == args.human else spec
                  for s in range(args.num_players)]

    try:
        Handler.game = Game(args)
    except Exception as e:
        ap.error(str(e) or repr(e))
    if args.human is not None:
        Handler.game.agent_turn()
    srv = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print('Splendor GUI: http://127.0.0.1:%d  (%s, %d players, %s)' % (
        args.port,
        'watch mode' if args.human is None else 'you are seat %d' % args.human,
        args.num_players,
        ' / '.join(spec_label(s) if s else 'you' for s in Handler.game.specs)),
        flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nbye')


if __name__ == '__main__':
    main()
