# Splendor for PufferLib 3.0

A fast, self-play Splendor environment: all game logic lives in one C header
(`splendor/splendor.h`), wrapped by a thin PufferLib binding and a thin Python
`PufferEnv`. Roughly 4M turns/s single-core with cached random actions and
~1.9M turns/s when sampling uniformly from the legal-action mask (M-series Mac).

One C env = one game of `num_players` seats; each seat is a PufferLib agent, so
`num_agents = num_envs * num_players` and training is pure self-play. One env
step = one Splendor turn: only the seat to move acts, the others are idle (mask
= PASS only, reward 0), so a masked policy gets zero gradient there.

## Rules

The real 2-4 player rules: 90 development cards in 3 tiers (4 face-up per
tier), 10 nobles of which `num_players + 1` are dealt, bank of 4/5/7 gems per
color (2/3/4 players) plus 5 gold. Each turn: take 3 different gems (or fewer,
as the official FAQ allows), take 2 of one color (only if 4+ are left),
reserve a card (face-up or deck top; +1 gold whenever the bank has one), or
buy a card (face-up or reserved) paying with bonuses, gems, then gold. If a
take or reserve leaves you above 10 tokens you must return tokens of your
choice down to 10, one per step. After a buy, a noble whose requirement you
meet visits you (at most one per turn); if several qualify you choose one.
Cards reserved from a deck top are hidden from opponents. Games end at the end
of the round in which someone reaches 15 points, so everyone gets the same
number of turns; the start seat is randomized per game. Winner = most points,
tiebreak fewest purchased cards, otherwise a draw. `max_turns` (total turns,
default 60 per seat) caps runaway games with the same scoring.

Sub-phases (discard, noble choice) are extra env steps by the same seat and do
not advance the turn counter. Illegal actions (per the mask) are treated as
PASS outside a sub-phase and as the first legal option inside one; they are
counted in `log.invalid` and carry no penalty. A masked policy never emits one.

## Install and build

```bash
conda create -n splendor python=3.12 && conda activate splendor
pip install pufferlib torch numpy pytest
pip install -e .          # builds splendor/binding.*.so
python -m pytest tests -q
```

After editing `splendor/splendor.h`, rebuild with
`python setup.py build_ext --inplace` (or `pip install -e .` again).

## Benchmark

```bash
python -m splendor.splendor     # turns/s and agent-steps/s, raw and legal actions
```

## Train

Config lives in `config/splendor.ini` (layered on top of PufferLib's
`config/default.ini`). Any key is overridable as `--section.key`.

```bash
python train.py                                   # device auto: cuda > mps > cpu
python train.py --train.device mps                # Mac
python train.py --train.device cuda --vec.num-envs 16 --vec.num-workers 16
python train.py --env.num-players 4 --rnn-name Recurrent   # 4p, LSTM policy
```

Checkpoints are written to `experiments/`.

## League training (the serious run)

`config/league.ini` trains the `Big` policy (3 x 512 LayerNorm MLP) inside a
league: the trainer only sees seat 0 of every game, the other seats are played
by frozen policies on CPU inside the workers. Half of the games are true
self-play against the latest weights (reloaded from `experiments/latest.pt`
every few seconds), the rest are against a pool of past checkpoints sampled
by prioritized fictitious self-play (opponents the learner loses to most are
sampled most). A checkpoint is added to `experiments/pool/` every
`pool_every` steps. Because idle seats never reach the trainer, every sample is
a real decision.

```bash
python train.py --config league                          # 2 players, 1B steps
python train.py --config league --env.num-players 3      # or 4
python train.py --config league --train.device cuda --vec.num-envs 16 --vec.num-workers 16
python train.py --config league --load-model-path experiments/latest.pt   # resume
python -m splendor.league                                # wrapper throughput
```

Dashboard user stats include `wr_latest` / `wr_pool` (learner win rates) and
`pool_size`. Note that a `Big` policy needs a few tens of millions of steps
before greedy play stops looping "take 1, discard 1" at the token cap.

## Evaluate and rate

```bash
python evaluate.py --load-model-path latest --opponent greedy --games 1000
python evaluate.py --load-model-path latest --opponent random --num-players 4
python evaluate.py --load-model-path latest --render         # one game, ANSI board
python elo.py --num-players 2 --games 200 --participants random greedy latest 'experiments/pool/*.pt'
python elo.py --num-players 3 --games 300 --participants random greedy 'experiments/pool/*.pt'
python elo.py --participants random greedy latest --mcts experiments/latest.pt:100
```

`elo.py` plays a round robin (2 players) or random tables (3-4 players) with
`splendor.agents.play_games`, fits Bradley-Terry ratings on the pairwise
results and prints them on the Elo scale with `random` anchored at 0, also
writing `elo.json`. `greedy` is a fixed heuristic yardstick (buy the most
valuable affordable card, else take gems toward the closest card): it beats
random 100% of the time and is roughly as strong as a 60M-step self-play
checkpoint, so it is a good early milestone. A checkpoint can only play at
the player count it was trained for (the observation width differs).

## MCTS and AlphaZero-style training

`splendor/mcts.py` is a batched PUCT search that uses the env's state
snapshots (`get_state` / `put_state`) and the policy's priors and values. It
evaluates every seat's observation at each leaf, so multiplayer backups are
exact, and it re-deals what the mover cannot see (deck order, opponents'
deck-top reserves) before searching, so it never peeks at hidden cards. About
26K simulations/s on CPU with the Big net at batch 64.

```bash
python -m splendor.mcts cpu                       # search benchmark
python elo.py --participants random greedy latest --mcts latest:64
python alphazero.py --num-players 2 --games 64 --sims 100 --iters 2 --init latest
python gui.py --load-model-path latest             # (MCTS in the GUI: see MCTSAgent)
```

`alphazero.py` runs the AlphaZero loop: self-play with search on every seat,
then train the same policy/value net on (observation, visit distribution,
outcome) samples, then repeat, saving `experiments/az/iter_k.pt` which loads
anywhere a checkpoint does. Warm-start it from a PPO checkpoint with
`--init latest`. Search with 64 simulations already beats the raw policy it
searches with about 63% of the time.

## GUI: watch the agent or play against it

`gui.py` serves a self-contained web page (Python stdlib only, no extra
dependencies): a full-screen square table with the board in the centre and a
mat per player around it (you at the bottom), purchased cards stacked by color
with their points, tokens, reserved cards and nobles. Every move is animated
from a structured event the server emits: bought and reserved cards fly to
the mover's mat, the replacement card flips in from the deck with a NEW tag,
gems fly between the bank and the mats. All art is inline SVG.

```bash
python gui.py --load-model-path latest                 # you are seat 0 vs the policy
python gui.py --load-model-path latest --human none    # watch mode: step button + auto-play slider
python gui.py --num-players 3 --human 2 --sample       # 3 players, you are seat 2, stochastic agents
python gui.py                                          # no checkpoint: random-legal opponents
```

Open the printed URL (default `http://127.0.0.1:8000`). Click gems to build a
take (1-3 different, or 2 of one color), click a face-up card for Buy /
Reserve, a deck for Reserve, or one of your reserved cards for Buy; only moves
that are legal per the action mask are enabled. Over 10 tokens, click one of
your own tokens to return it; when several nobles qualify, click one.

Everything is also on the keyboard: `1`-`5` pick gems (twice for two of a
color, and in the discard / noble phases they return a token or choose a
noble slot; `6` returns gold), the arrow keys move a card cursor over the
grid (tier 3 / 2 / 1, then your reserved cards, with the deck in column 0),
`Enter` takes the selected gems or buys the focused card (reserving it when it
cannot be bought), `B` / `R` buy / reserve, `Esc` clears the selection and the
cursor, `P` passes when passing is the only legal move, `N` deals a new game,
`Space` steps in watch mode, `T` opens the table picker and `?` shows the key
list. A key never fires an action the legal mask forbids.

The game ends at the end of the round in which someone reaches 15 points (or
at the turn limit). The finished board is **held**: every card, token, mat and
reserved card stays exactly as it ended, polling and stepping stop, and a
results panel docks above the action bar with the standings (rank, seat,
agent, points, cards, nobles, tokens left), how the game ended and the turn
count. "Hide results" collapses it to a strip so the whole table is visible;
"New game" (or `N`) deals the next one.

The **Table** button opens the opponent picker: number of players, your seat
(or watch), and per seat `random`, `greedy`, any checkpoint under
`experiments/` (filtered to the chosen player count) with an optional MCTS
search and simulation count, plus a sample-moves toggle. `--mcts SIMS` on the
command line puts search on every agent seat from the start.

## Observations and actions

`Box(uint8, (240 + 48*P,))` - 336 / 384 / 432 bytes for 2 / 3 / 4 players,
perspective-relative (me first, then the others in turn order). A card is 11
bytes: `cost[5], bonus one-hot[5], points[1]`.

| offset | size | content |
|---|---|---|
| 0 | 6 | bank: white, blue, green, red, black, gold |
| 6 | 3 | deck sizes, tiers 1-3 |
| 9 | 132 | 12 face-up cards (slot = tier*4 + pos) |
| 141 | 25 | 5 noble requirement vectors (zeros if absent/taken) |
| 166 | 48*P | per player: tokens[6], bonuses[5], points[1], 3 x (card + present) |
| 166+48P | 72 | legal action mask for this seat |
| 238+48P | 1 | turn number, clipped to 255 |
| 239+48P | 1 | 1 iff it is this seat's move (also when it can only pass) |

`Discrete(72)`: 0-9 take 3 distinct colors, 10-19 take 2 distinct, 20-24 take
1, 25-29 take 2 of one color, 30-41 reserve a face-up slot, 42-44 reserve a
deck top, 45-56 buy a face-up slot, 57-59 buy one of your reserved cards, 60
pass (legal only when nothing else is, or the seat is idle), 61-66 return one
token of that color (gold = 66) while over 10 tokens, 67-71 choose the dealt
noble in that slot when several qualify.

All offsets, action ranges and the policy's input-scaling table live in
`splendor/layout.py`, which the tests, the policy and `evaluate.py` share.

## Policy

`splendor/policy.py`: the observation is scaled per index into roughly [0, 1],
then a 2-layer GELU MLP (hidden 256) feeds an actor and a value head. The mask
slice is read out of the observation in `encode_observations` and applied in
`decode_actions` as `torch.where(mask, logits, -1e8)`, so illegal actions are
never sampled and entropy stays finite. `Recurrent` is the PufferLib
`LSTMWrapper` around it.

`Splendor(auto_reset=False)` holds a finished board (all-zero legal mask,
steps are no-ops) until `reset()`; training uses the default `auto_reset=True`,
which deals the next game inside the terminal step.
`env.get_state(i)` returns an opaque snapshot of game `i` and
`env.put_state(i, state=...)` restores it (useful for search / MCTS);
`env.put_state(i, bonuses=[...])` overwrites bonus cards for tests.
