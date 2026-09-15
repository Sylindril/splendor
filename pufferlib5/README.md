# Splendor for PufferLib 5.0

PufferLib 5.0 is a pure C/CUDA rewrite: an environment is a single header that
is compiled directly into the trainer. This directory is the 5.0 front-end for
the Splendor environment in this repo.

| file | what it is |
|---|---|
| `splendor.h` | the 5.0 env: `puf_init/reset/step/render/close/log`, the bot ladder, a raylib board |
| `splendor.ini` | trainer config, layered on PufferLib's `config/default.ini` |
| `install.sh` | copies `splendor.h` + `../splendor/game.h` + `splendor.ini` into a 5.0 checkout |

All the rules, tables, observations and legal-action masks live in
`../splendor/game.h`, shared verbatim with the PufferLib 3.0 binding
(`../splendor/splendor.h`). The two front-ends cannot drift apart.

## Install and build

```bash
./pufferlib5/install.sh /path/to/PufferLib     # the 5.0 checkout with build.sh
cd /path/to/PufferLib

./build.sh splendor && ./puffer train splendor   # training: needs CUDA + nvcc
NVCC_EXTRA="-DSPLENDOR_PLAYERS=4" ./build.sh splendor puffer4   # 4 seats
./build.sh splendor --cpu && ./splendor --headless --eval_episodes=100
./splendor latest                                # watch the newest checkpoint
```

**Training requires CUDA/nvcc**: `./build.sh splendor` compiles `src/pufferl.cu`
with nvcc and links cuBLAS/NCCL. There is no CPU trainer. The `--cpu` build is
clang + raylib + libomp only and gives you the play/eval binary, which runs on
a Mac. On macOS run `build.sh` under a bash 4+ (`bash ./build.sh splendor
--cpu`): the script uses `${ENV^^}`, and `/bin/bash` is still 3.2.

The seat count is a compile-time constant (`SPLENDOR_PLAYERS`, default 2, range
2..4) because `OBS_SIZE` must be constant. `[vec] total_agents` must be a
multiple of it: 8192 is fine for 2 and 4 seats, use 8190 for 3.

## The environment

One `Env` is one game of `SPLENDOR_PLAYERS` seats and one `Agent` per seat.
One `puf_step` is one action by the seat to move; the other seats are idle that
step. Games always auto-reset, so the observations returned by a terminal step
already belong to the next game.

- **Observation** `uint8[OBS_SIZE]`, `OBS_SIZE = 240 + 48 * SPLENDOR_PLAYERS`
  (336 / 384 / 432). Perspective-relative, "me" first: bank(6), deck sizes(3),
  12 face-up cards x 11, 5 nobles x 5, then 48 bytes per player (tokens,
  bonuses, points, 3 reserved cards), the 72-byte legal mask, a turn counter
  and a to-move flag. Opponents' deck-top reserves are hidden.
- **Action** `Discrete(72)`: take 3 different / 2 different / 1 / 2 of a kind,
  reserve a face-up card or a deck top, buy a face-up or reserved card, pass,
  plus the two sub-phases (return a token when over 10, choose among several
  qualifying nobles). See DESIGN.md for the exact table.
- **Masks** are written for every seat every step. An idle seat's mask is PASS
  only, so a masked policy has zero gradient there.
- **Rewards** `reward_point * points_gained + reward_card * cards_bought` at
  the acting seat's own step, plus `reward_win` / `reward_loss` for everyone at
  the end. A draw among the tied leaders pays them 0.
- **Log**: `perf` (fraction of games seat 0 wins, a draw counts 0.5), `score`
  (winner's points), `points`, `cards`, `nobles`, `game_length` (turns),
  `invalid` (illegal actions submitted), `n` (finished games).

`[env]` keys: `max_turns` (0 = 60 turns per seat), `reward_point`,
`reward_card`, `reward_win`, `reward_loss`, `bot_policy`.

## Self-play and the bot ladder

`[selfplay] enabled = 1` with `[vec] num_policies = 2` trains seat 0 against
frozen checkpoints from the same run on the `hist_policy_percent` tail of the
envs; on the rest every seat is the live policy.

`bot_policy` replaces seats > 0 with a scripted opponent inside `puf_step`:
`0` none, `1` uniform over legal actions, `2` a greedy heuristic (buy the
affordable card worth the most points, else take the gems that most reduce the
shortfall of the closest card, else reserve that card; discard the colour the
closest card needs least; take the first qualifying noble). It is the C port of
`splendor/agents.py:GreedyAgent`. Use it with `[selfplay] eval_bots = 1,2` and
`eval_bot_games > 0`, or straight from the CPU binary:

```bash
./splendor --headless --eval_episodes=100 --env.bot_policy=2
```

## Using a trained 5.0 policy in the Python tools

`./puffer train splendor` exports `checkpoints/splendor/..._weights.bin`: flat
float32, no biases, in order encoder `(H, OBS_SIZE)`, decoder `(73, H)`, then
one `(3H, H)` MinGRU projection per layer, each block padded to a multiple of 8
floats. `splendor/puffernet.py` reproduces that forward pass in torch, so the
repo's evaluation tools take a 5.0 checkpoint as an agent spec:

```bash
python evaluate.py --load-model-path puffer5:checkpoints/splendor/xxx_weights.bin:512:2
python elo.py --participants random greedy puffer5:PATH_TO_weights.bin:512:2
```

The spec is `puffer5:PATH:HIDDEN:LAYERS`, with `HIDDEN` / `LAYERS` matching
`[policy] hidden_size` / `num_layers` of the run that produced it. The GUI's
model picker lists `*_weights.bin` files the same way.

## Hyperparameters

`[train]` in `splendor.ini` is **untuned** — PufferLib's defaults with a longer
budget and a shorter horizon. Sweep (`./puffer sweep splendor`) before reading
anything into a learning curve.
