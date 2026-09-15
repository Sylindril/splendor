# Splendor PufferLib environment: design spec

Target: PufferLib 3.0 (installed at `~/miniforge3/envs/splendor/lib/python3.12/site-packages/pufferlib`),
Python 3.12 in conda env `splendor` (`~/miniforge3/envs/splendor/bin/python`). Torch 2.14 (MPS on this Mac, CUDA on servers).

Goal: small, fast, simple. Game logic in a single C header, thin PufferLib binding, thin Python wrapper,
masked MLP policy, one train script. No raylib, no external deps beyond pufferlib/numpy/torch.

## Layout

```
Splendor/
  splendor/__init__.py       # exports Splendor env class
  splendor/splendor.h        # ALL game logic (cards, nobles, rules, obs, mask, step, reset, ansi render)
  splendor/binding.c         # pufferlib glue: my_init / my_log only (uses env_binding.h from pufferlib/ocean)
  splendor/splendor.py       # PufferEnv wrapper + __main__ SPS benchmark
  splendor/policy.py         # Policy (masked MLP) and Recurrent (LSTM wrapper)
  splendor/layout.py         # pure-python constants: obs offsets, action tables (shared by tests/policy/eval)
  config/splendor.ini        # trainer config (same format as pufferlib/config/*.ini)
  train.py                   # python train.py [--train.device mps] [--env.num_players 2] ...
  evaluate.py                # policy vs random-legal opponent win rate; optional ansi render
  setup.py                   # builds splendor/binding.*.so;  pip install -e .
  tests/test_splendor.py     # pytest
  README.md
```

## Game rules (exact 2-4 player Splendor; the three simplifications below were REMOVED on 2026-09-11, see "Exact rules")

Colors: 0 white, 1 blue, 2 green, 3 red, 4 black; 5 gold (joker).
Bank per color: 4 (2 players), 5 (3), 7 (4); gold always 5.
Cards: 90 standard cards. Tier 1: 40 (8 per color: seven 0-pt, one 1-pt), tier 2: 30 (6 per color, pts 1,1,2,2,2,3),
tier 3: 20 (4 per color, pts 3,4,4,5). 4 face-up cards per tier, refilled from that tier's deck after a card leaves.
Nobles: the 10 standard nobles (3 points each, requirement = bonus cards); `num_players + 1` are dealt.

Turn actions (exactly one per turn):
- Take 3 gems of different colors, 2 of different colors, or 1 gem: legal iff each color has >= 1 in bank
  and player's token total after taking <= 10.  (Simplification 1: taking 1 or 2 different is always allowed;
  the "take then discard down to 10" rule is replaced by "a take is only legal if it keeps you at <= 10".)
- Take 2 gems of one color: legal iff bank has >= 4 of that color and total after <= 10.
- Reserve a face-up card or the top card of a deck: legal iff reserved count < 3 (and slot/deck non-empty).
  Player gets 1 gold iff bank gold >= 1 AND player token total < 10.  (Simplification 2.)
- Buy a face-up card or one of own reserved cards: legal iff affordable.
  Affordable: for each color c, need = max(0, cost[c] - bonus[c]); short = max(0, need - tokens[c]);
  sum(short) <= gold tokens. Payment: pay tokens first, then gold for the shortfall. Tokens go back to bank.
- Pass: legal ONLY if no other action is legal (and for idle seats, see below).
After a buy, nobles are checked: the first noble (in dealt order) whose requirement is met is taken automatically
(+3 points), removed from the table. At most one noble per turn. (Simplification 3: no choice among nobles.)
Reserved cards taken from a deck top are hidden from opponents; face-up reserves are visible.

End: at the end of a round (after the last seat in turn order acts) if any player has >= 15 points, the game ends.
Also ends when total turns >= max_turns. Winner = most points; tiebreak = fewest purchased cards; still tied = draw
among the tied players. Starting seat is randomized per game so seat index carries no advantage.
Turn order: seat (start + t) % P. Last seat of a round = (start + P - 1) % P.

## Exact rules (supersedes the simplifications above)

- Takes are limited only by the bank. If a take or reserve leaves the player above 10 tokens, the env enters
  PHASE_DISCARD: the same seat keeps acting, only actions 61-66 ("return one token of color c", gold = 66) are legal,
  one token per step, until it is at <= 10. Reserving always grants a gold when the bank has one.
- After a buy, if exactly one dealt noble qualifies it is taken automatically; if several qualify the env enters
  PHASE_NOBLE and only actions 67-71 ("choose dealt noble slot i") are legal for the qualifying slots.
- Sub-phase steps do not increment `turn`; the end-of-round check runs only when the turn completes.
  Invalid actions inside a sub-phase are forced to the first legal option (never a stall).
- Action space is Discrete(72); OBS_N = 240 + 48P (mask 72 wide, then turn byte, then a to-move byte that is
  1 iff the env is waiting on this seat; `layout.to_move(obs, P)`). A stalled game (bank drained, 3 reserves each,
  nothing affordable) has a pass-only mask for the seat to move; it ends via max_turns. `layout.acting(mask)` = any legal action
  other than PASS identifies the seat to move (it may be in a sub-phase).
- binding exposes `env_get(handle) -> {'state': bytes}` and `env_put(handle, state=bytes | bonuses=[P*5 ints])`
  (Python: `env.get_state(i)`, `env.put_state(i, ...)`), recomputing mask + obs.

## Agents / self-play

One C `Splendor` struct = one game with P players (P = num_players, 2..4). Each seat is a PufferLib agent.
Python: `num_agents = num_envs * num_players`. Game i owns agent rows `[i*P, (i+1)*P)` of every buffer; the
Python wrapper calls `binding.env_init(obs[i*P:(i+1)*P], actions[i*P:(i+1)*P], rewards[...], terminals[...],
truncations[...], seed_i, **kwargs)` per game and then `binding.vectorize(*handles)` (same pattern as
pufferlib/ocean/moba/moba.py). One env step = ONE turn: only the current seat's action is applied; other seats'
actions are ignored. Idle seats get reward 0 and a mask with only PASS legal (so a masked policy has zero gradient there).
Every step, observations for ALL P seats are rewritten (state changed), rewards[all]=0 first, terminals[all]=0.

Invalid action (illegal per mask): treated as PASS, counted in log.invalid. No penalty.

Rewards (per acting seat, at its own step): `reward_point * points_gained + reward_card * cards_bought`.
At game end: all seats get terminal=1 and reward += reward_win (+1) for the winner(s), reward_loss (-1) for losers.
Draw among tied top players: 0 for them, -1 for the others. Then the game is immediately reset (new game obs written
into the same step's observations, as all ocean envs do). Trainer clamps rewards to [-1, 1].

## Actions: Discrete(61)

| range | meaning | legality |
|---|---|---|
| 0-9 | take 3 distinct colors, combos in lexicographic order: (0,1,2),(0,1,3),(0,1,4),(0,2,3),(0,2,4),(0,3,4),(1,2,3),(1,2,4),(1,3,4),(2,3,4) | all 3 colors in bank >= 1, tokens+3 <= 10 |
| 10-19 | take 2 distinct: (0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4) | both >= 1, tokens+2 <= 10 |
| 20-24 | take 1 of color c = a-20 | bank[c] >= 1, tokens+1 <= 10 |
| 25-29 | take 2 same of color c = a-25 | bank[c] >= 4, tokens+2 <= 10 |
| 30-41 | reserve face-up slot s = a-30 (tier = s/4, pos = s%4) | slot non-empty, reserved < 3 |
| 42-44 | reserve top of deck tier t = a-42 | deck non-empty, reserved < 3 |
| 45-56 | buy face-up slot s = a-45 | non-empty, affordable |
| 57-59 | buy own reserved card index r = a-57 | exists, affordable |
| 60 | pass | only if nothing else legal, or seat is idle |

`layout.py` must expose these as constants and helper tables: `NUM_ACTIONS=61`, `TAKE3`, `TAKE2`, `TAKE1`, `TAKE2SAME`,
`RESERVE_FACEUP`, `RESERVE_DECK`, `BUY_FACEUP`, `BUY_RESERVED`, `PASS` (start indices), `COMBOS3`, `COMBOS2` lists.

## Observation: Box(uint8, shape=(OBS_N,)), OBS_N = 228 + 48*P  (P=2: 324, 3: 372, 4: 420)

Perspective-relative: "me" first, then the other players in turn order after me ((me+1)%P, (me+2)%P, ...).
Card encoding CARD = 11 bytes: cost[5] (w,b,g,r,k), bonus one-hot[5], points[1]. Empty slot = 11 zeros.

| offset | size | content |
|---|---|---|
| 0 | 6 | bank tokens: white, blue, green, red, black, gold |
| 6 | 3 | deck sizes tier 1,2,3 |
| 9 | 132 | 12 face-up cards x CARD, slot s = tier*4 + pos |
| 141 | 25 | 5 nobles x requirement[5]; unused/taken noble slots are zeros |
| 166 | 48*P | players, me first. Per player 48 bytes: tokens[6], bonuses[5], points[1], 3 reserved x 12 (CARD + present flag). Opponent's hidden reserved card: present=1, other 11 bytes 0 |
| 166+48P | 61 | legal action mask (1 = legal) for this seat; idle seats: only index 60 set |
| 227+48P | 1 | turn number (total turns so far, clipped to 255) |

`layout.py` exposes `obs_size(P)`, and offsets `BANK, DECKS, FACEUP, NOBLES, PLAYERS, mask_offset(P), turn_offset(P)`,
`CARD_N=11`, `PLAYER_N=48`, `NOBLE_MAX=5`.

## C API (splendor.h)

```c
typedef struct { float score; float points; float cards; float nobles; float game_length;
                 float perf; float invalid; float n; } Log;   // floats only, n LAST (required by env_binding.h)
typedef struct {
    Log log;                       // required first
    unsigned char* observations;   // P rows of OBS_N
    int* actions;                  // P ints
    float* rewards;                // P floats
    unsigned char* terminals;      // P bytes
    int num_players, max_turns;
    float reward_point, reward_card, reward_win, reward_loss;
    uint64_t rng;                  // per-env xorshift/splitmix RNG seeded from kwargs "seed"; NO rand()
    ... game state ...
} Splendor;
void c_reset(Splendor* env);   // new game (shuffle decks, deal, random start seat), writes all obs
void c_step(Splendor* env);    // one turn, as described above
void c_render(Splendor* env);  // printf an ANSI text board; no raylib
void c_close(Splendor* env);   // nothing to free (no heap allocations inside the env)
```
Log semantics, accumulated once per finished game: score = winner's points, points = mean final points over seats,
cards = mean cards bought per seat, nobles = mean nobles per seat, game_length = turns, perf = 1.0 if ended by
reaching 15 points else 0.0 (hit max_turns), invalid = invalid actions in the game, n += 1.

binding.c:
```c
#include "splendor.h"
#define Env Splendor
#include "env_binding.h"          // from pufferlib/ocean (include dir set in setup.py)
static int my_init(Env* env, PyObject* args, PyObject* kwargs) { num_players, max_turns, reward_point, reward_card,
    reward_win, reward_loss, seed via unpack(kwargs, ...); init rng; return 0; }
static int my_log(PyObject* dict, Log* log) { assign_to_dict for every field except n; return 0; }
```
Card table: `static const` array of 90 cards `{tier, bonus, points, {w,b,g,r,k}}` and 10 nobles `{w,b,g,r,k}`.
Decks: fixed arrays `unsigned char deck[3][40]; int deck_n[3]` of card ids, Fisher-Yates shuffled on reset.
Face-up: `short faceup[3][4]` card id or -1. Reserved: per player `short reserved[3]` + `unsigned char hidden[3]`.
Cache the legal mask for the current seat in the struct so c_step doesn't recompute it.
Performance target: > 1M turns/s single core on this Mac with random actions. Keep everything on the stack/inline,
no malloc per step, no printf in step.

## Python env (splendor/splendor.py)

```python
class Splendor(pufferlib.PufferEnv):
    def __init__(self, num_envs=1024, num_players=2, max_turns=None (-> 60*num_players), reward_point=0.02, reward_card=0.0,
                 reward_win=1.0, reward_loss=-1.0, report_interval=128, render_mode=None, buf=None, seed=0)
```
`single_observation_space = Box(0, 255, (obs_size(P),), uint8)`, `single_action_space = Discrete(61)`,
`num_agents = num_envs * num_players`. reset/step/render/close follow pufferlib/ocean/moba/moba.py and
connect4.py (vec_log every report_interval steps appended to info only if it has keys). `__main__` runs an SPS
benchmark with random actions (report turns/s = num_envs * steps / time, and agent-steps/s).

## Policy (splendor/policy.py)

`class Policy(nn.Module)`: input = full obs as float times a per-index scale buffer (bank,cost 1/7; deck 1/40;
points 1/5 for cards, 1/20 for players; player tokens/bonuses 1/10; nobles 1/4; turn 1/max(255); mask 1.0).
MLP: Linear(OBS_N, hidden) GELU Linear(hidden, hidden) GELU. `hidden_size` attr, `is_continuous=False`.
`encode_observations(obs, state=None)` stashes the mask slice as `self._mask` (bool) and returns hidden;
`decode_actions(hidden)` -> `logits = actor(hidden); logits = torch.where(self._mask, logits, -1e8)`; value head.
`forward_eval`/`forward` as in pufferlib.models.Default. `Recurrent = pufferlib.models.LSTMWrapper` subclass with
input_size=hidden_size=hidden. Default hidden 256. Masked with -1e8 (not -inf) so entropy stays finite.

## train.py / config

Mimic pufferlib.pufferl.load_config but read `[pufferlib/config/default.ini, config/splendor.ini]` and add
`--section.key` CLI overrides with identical nesting (copy that ~40-line block). Device default 'auto' resolved in
train.py to cuda > mps > cpu. Build `vecenv = pufferlib.vector.make(Splendor, env_kwargs=args['env'], **args['vec'])`,
policy from splendor.policy by `policy_name`/`rnn_name`, then `pufferlib.pufferl.train('puffer_splendor', args=args,
vecenv=vecenv, policy=policy)`. Also `python train.py eval --load-model-path ...` -> delegate to evaluate.py logic.
`config/splendor.ini`: `[base] package = splendor, env_name = puffer_splendor, policy_name = Policy, rnn_name = None`
`[env] num_envs = 512, num_players = 2`; `[vec] backend = Multiprocessing, num_envs = 8, num_workers = 8, batch_size = 4`
(so the driver gets 4 workers' batches at a time); `[train] total_timesteps = 100_000_000, bptt_horizon = 16,
minibatch_size = 16384, device = auto`, other keys inherited from default.ini. Keep it short.

## gui.py (added later)

Zero-dependency local web GUI: `python gui.py [--load-model-path X] [--human 0|none] [--num-players P] [--port 8000] [--rnn] [--sample]`.
stdlib http.server + `splendor/gui/index.html`; decodes the board from the observation; human seat clicks map to action indices via layout.py.

## evaluate.py

`python evaluate.py --load-model-path experiments/xxx.pt [--games 1000] [--render]`: single-process
`Splendor(num_envs=games)`; seat 0 = policy (greedy argmax over masked logits), other seats = uniform random over
legal actions from the obs mask. Because start seat is random, play until every game has finished once; report policy
win rate, draw rate, mean points. With --render, num_envs=1 and print `env.render()` each turn.

## Tests (tests/test_splendor.py), run with `~/miniforge3/envs/splendor/bin/python -m pytest tests -q`

1. Card table sanity via a tiny debug entry: expose card/noble tables to Python through `binding.shared()`?
   Simpler: tests read the face-up cards from observations over many resets and check structural facts, AND
   layout.py is imported to check obs_size. Additionally the C header must be self-consistent: a test compiles nothing.
2. Random legal play (sample uniformly from the mask, num_envs=256, 20k steps): never crashes; per game
   token conservation from obs (bank + all players' tokens == initial totals, per color, incl gold); every step
   at most one seat has a non-pass-only mask (the actor can be pass-only when the bank is drained and nothing is affordable); terminals arrive; log dict has expected keys;
   game_length <= max_turns.
3. Scripted scenarios from a fresh game with num_envs=1, seed fixed: take-3 changes bank and my tokens by exactly
   the combo; take-2-same illegal when bank < 4; reserve deck top gives one gold and decrements deck size;
   idle seat mask is pass-only; invalid action does not change state; buying a card after collecting tokens
   increments bonus and refills the slot.

# Phase 2 (2026-09-11): league self-play, Elo, MCTS infrastructure

Shared pieces that already exist (do not redesign them):
- `splendor/agents.py`: `Agent.act(obs, mask, env=None, games=None) -> int32 actions`, `Agent.reset(games=None)`,
  `RandomAgent`, `PolicyAgent(policy, device, temperature, name)` / `PolicyAgent.from_checkpoint(path, num_players, device)`,
  `load_policy(path, num_players, device)` (infers Policy/Big architecture from the state dict; 'latest' = newest .pt under
  experiments/), `play_games(agents, num_games, num_players, seed, max_turns, env, verbose)` -> dict(winner (-1 = draw),
  points (G,P), turns, ranks (G,P; 0 = best, ties share), finished), `win_rates(result, P)`.
  play_games sends PASS to every seat by default and the agent's action to the seat to move (`layout.to_move`).
- `splendor/policy.py`: `Policy(env, hidden_size, layers, norm)`, `Big` (512 x 3, LayerNorm), `Recurrent`,
  `arch_from_state_dict(sd)`.
- Env: `env.get_state(i) -> bytes`, `env.put_state(i, state=bytes | determinize=seat | bonuses=[...])`; action
  `layout.NOOP (-1)` for the seat to move freezes that game for the step; obs byte `layout.to_move_offset(P)`.
- `layout.acting(mask)` = some legal action other than PASS; `layout.to_move(obs, P)` = the env waits on this seat.

## splendor/league.py (owner: league agent)
`class League(pufferlib.PufferEnv)`: same spaces as Splendor, `num_agents = num_envs` (the learner is seat 0 of every
game; the env randomizes the start seat). Wraps an inner `Splendor(num_envs, num_players, ...)` with its own buffers.
kwargs: num_envs=512, num_players=2, pool_dir='experiments/pool', latest_path='experiments/latest.pt', latest_frac=0.5,
pfsp_exponent=2.0, reload_interval=500 (wrapper steps between checks of latest.pt mtime / pool listing), max_turns=None,
reward_point=0.02, reward_card=0.0, reward_win=1.0, reward_loss=-1.0, report_interval=128, render_mode=None, buf=None, seed=0.
Opponents: torch CPU models (`torch.set_num_threads(1)`, inference_mode) loaded with `agents.load_policy`; 'latest'
reloaded when latest_path's mtime changes; pool = sorted .pt files in pool_dir. If latest.pt is missing, opponents are
random-legal. Per game, at game start, each opponent seat independently gets: with prob latest_frac the latest model,
else a pool member sampled with PFSP weights `(1 - w_i)**pfsp_exponent + 0.05` where w_i is an EMA (0.99) of the
learner's win rate vs member i (init 0.5); empty pool -> latest.
`step(actions)`: repeat { for each game whose to-move seat is 0: use the learner's action if not consumed yet this call
else NOOP; for games whose to-move seat is an opponent: batched opponent inference grouped by model; step inner;
accumulate the learner's reward (sum) and terminal (or) into the wrapper buffers; on a terminal, record win/draw/loss
against that game's opponents and resample opponents } until every game waits on seat 0 (cap 8*P+8 iterations; leftover
games are simply handled on the next call, the learner's action for them is dropped, which is harmless since its obs
said not-to-move). Then copy inner obs rows 0::P into self.observations. Games whose to-move seat can only pass (stalled)
get PASS like any other move. `info` = inner vec_log dict every report_interval steps, plus league keys
`pool_size`, `wr_latest`, `wr_pool` (learner win rates, EMA) when available.

## train.py additions (owner: league agent)
`--config NAME` (default 'splendor') chooses `config/NAME.ini`; `[base] env_class = Splendor|League`.
`config/league.ini`: `[base] env_class = League, policy_name = Big`; `[env] num_envs = 512, num_players = 2,
latest_frac = 0.5, pool_dir = experiments/pool`; `[vec]` like splendor.ini; `[league] save_every = 30` (seconds between
atomic writes of latest.pt: torch.save to a tmp file then os.replace), `pool_every = 20_000_000` (global steps between
copies of latest.pt into pool/step_{step:011d}.pt); `[train] total_timesteps = 1_000_000_000, bptt_horizon = 16,
minibatch_size = 32768`. The league loop is a copy of pufferlib.pufferl.train's while loop with those two saves added.
All player counts must work: `--env.num-players 3`.

## elo.py + GreedyAgent + evaluate.py (owner: elo agent)
`agents.GreedyAgent`: a fixed heuristic yardstick (buy the affordable card with most points, tiebreak higher tier;
else take gems that most reduce the shortfall of the closest-to-affordable face-up or reserved card; else reserve the
best card if that is legal and useful; discard the most-held color the closest card doesn't need; first noble).
`python elo.py --num-players P --games N --participants random greedy experiments/pool/*.pt latest [--mcts path:sims]
[--device cpu] [--out elo.json]`: 2 players = round robin, every pair plays N games (seat order swapped halfway);
3-4 players = N random tables of P distinct participants. Each game yields pairwise results from `ranks`
(lower rank wins, equal = draw). Ratings by Bradley-Terry (iterative MM or logistic fit) on the pairwise
win/draw counts, reported on the Elo scale with `random` anchored at 0. Prints a sorted table and writes JSON.
`evaluate.py` becomes a thin front-end over `play_games` (policy vs random / greedy / checkpoint, any P, --render).

## splendor/mcts.py + alphazero.py (owner: mcts agent)
`class MCTS(policy, num_players, sims=200, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, determinize=True,
device='cpu', seed=0)`: batched PUCT over a batch of root states (bytes from env.get_state). Uses a scratch
`Splendor(num_envs=batch)` for transitions via put_state / step / get_state. Node values are per-seat vectors (the net
is evaluated on every seat's obs row of the leaf state, so multiplayer backups are exact); terminal leaves use the env
rewards. Selection uses Q from the to-move seat's perspective. Root: optional Dirichlet noise; `determinize=True`
re-deals what the to-move seat cannot see before the snapshot. The env RNG is part of the snapshot, so a search is
one determinization (document this). `search(states) -> (visit_counts (B, 72), values (B, P))`.
`class MCTSAgent(Agent)`: act(obs, mask, env, games) snapshots those games, searches, returns argmax visits (or a
temperature sample). `alphazero.py`: self-play with MCTSAgent on every seat (temperature 1.0 for the first 15 turns,
then greedy), records (obs, visit distribution, final outcome per seat) for every decision, trains Policy/Big with
policy cross-entropy + value MSE for a few epochs, saves experiments/az/iter_{k}.pt (loadable with agents.load_policy),
repeats; `--init latest` warm-starts from a PPO checkpoint. Report sims/s.

- 2026-09-14: `auto_reset` kwarg (default 1). With 0 the env sets `game_over`, writes the final obs with an all-zero mask and ignores steps until c_reset (used by gui.py for an inspectable end screen).
