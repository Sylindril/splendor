// Splendor for PufferLib 5.0. One Env = one game of SPLENDOR_PLAYERS seats,
// one Agent per seat, one puf_step = one action by the seat to move. All the
// rules, tables, observations and legal masks live in game.h (shared with the
// PufferLib 3.0 binding); this file is buffer plumbing, rewards, the scripted
// bot ladder and a raylib board.
//
//   ./build.sh splendor && ./puffer train splendor          # CUDA trainer
//   NVCC_EXTRA="-DSPLENDOR_PLAYERS=4" ./build.sh splendor puffer4
//   ./build.sh splendor --cpu && ./splendor latest          # watch a game
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "raylib.h"
typedef unsigned char obs_t;
// Scripted opponents for [selfplay] eval_bots: 1 = random legal, 2 = greedy.
// Must be set before pufferenv.h, which stubs puf_set_bot_policy without it.
#define PUF_HAS_BOT_POLICY
#include "pufferenv.h"
#include "game.h"

// Seats per game, fixed at compile time because OBS_SIZE must be a constant.
#ifndef SPLENDOR_PLAYERS
#define SPLENDOR_PLAYERS 2
#endif
#if SPLENDOR_PLAYERS < 2 || SPLENDOR_PLAYERS > 4
#error "SPLENDOR_PLAYERS must be 2, 3 or 4"
#endif

#define ACT_SIZES {NUM_ACTIONS}
#define NUM_ATNS 1
#define OBS_SIZE GAME_OBS_SIZE(SPLENDOR_PLAYERS)
// One puf_step is one seat's action, so a 2p turn is 2 steps. 6/s is a
// watchable pace for the --cpu viewer.
#define PUF_STEPS_PER_SEC 6

struct Log {
    float perf;         // fraction of games seat 0 wins (a draw counts 0.5)
    float score;        // winner's points
    float points;       // mean final points over seats
    float cards;        // mean cards bought per seat
    float nobles;       // mean nobles per seat
    float game_length;  // turns played
    float invalid;      // illegal actions submitted during the game
    float n;            // finished games; keep last
};

struct Env {
    Log log;
    Agent agents[SPLENDOR_PLAYERS];
    int tag;
    int boundary_reached;
    int num_agents;
    // Seeded from the env index by pufferl's env_setup; drives the bots and,
    // through seed_rng, the game's own 64-bit stream.
    unsigned int rng;

    float reward_point;
    float reward_card;
    float reward_win;
    float reward_loss;
    int bot_policy;

    Game game;
};
typedef Env Splendor;

// Uniform over the legal actions of the seat to move.
static int bot_random(Splendor* env) {
    int legal[NUM_ACTIONS];
    int n = 0;
    for (int a = 0; a < NUM_ACTIONS; a++) {
        if (env->game.mask[a]) {
            legal[n++] = a;
        }
    }
    if (n == 0) {
        return ACT_PASS;
    }
    return legal[rand_r(&env->rng) % n];
}

// Token delta of take action a (0 .. ACT_TAKE2SAME + 4).
static void take_delta(int a, int* delta) {
    memset(delta, 0, NUM_COLORS * sizeof(int));
    if (a < ACT_TAKE2) {
        for (int i = 0; i < 3; i++) {
            delta[COMBOS3[a - ACT_TAKE3][i]] = 1;
        }
    } else if (a < ACT_TAKE1) {
        for (int i = 0; i < 2; i++) {
            delta[COMBOS2[a - ACT_TAKE2][i]] = 1;
        }
    } else if (a < ACT_TAKE2SAME) {
        delta[a - ACT_TAKE1] = 1;
    } else {
        delta[a - ACT_TAKE2SAME] = 2;
    }
}

#define NUM_TAKES (ACT_TAKE2SAME + NUM_COLORS)
#define NUM_BUYS (12 + MAX_RESERVED)
#define SHORT_FAR 1000  // shortfall of an empty / absent card slot

// The fixed heuristic yardstick ported from splendor/agents.py GreedyAgent:
// buy the affordable card worth the most points; else take the gems that most
// reduce the shortfall of the card closest to affordable; else reserve that
// card; else any legal take. Discard whatever the closest card needs least,
// take the first qualifying noble. It reads the real card tiers instead of
// guessing them from the cost, which the Python version has to do because it
// only sees an observation.
static int bot_greedy(Splendor* env) {
    Game* g = &env->game;
    int p = g->current;
    const unsigned char* mask = g->mask;

    if (g->phase == PHASE_NOBLE) {
        for (int i = 0; i < NOBLE_MAX; i++) {
            if (mask[ACT_NOBLE + i]) {
                return ACT_NOBLE + i;
            }
        }
        return ACT_PASS;
    }

    // Candidate cards in buy-action order: 12 face-up slots then own reserves.
    int card[NUM_BUYS];
    for (int s = 0; s < 12; s++) {
        card[s] = g->faceup[s / 4][s % 4];
    }
    for (int r = 0; r < MAX_RESERVED; r++) {
        card[12 + r] = g->reserved[p][r];
    }

    int gold = g->tokens[p][GEM_GOLD];
    int need[NUM_BUYS][NUM_COLORS];
    int shortfall[NUM_BUYS];
    for (int i = 0; i < NUM_BUYS; i++) {
        shortfall[i] = SHORT_FAR;
        memset(need[i], 0, sizeof(need[i]));
        if (card[i] < 0) {
            continue;
        }
        int deficit = 0;
        for (int c = 0; c < NUM_COLORS; c++) {
            int want = (int)CARDS[card[i]].cost[c] - (int)g->bonuses[p][c];
            need[i][c] = want > 0 ? want : 0;
            int missing = need[i][c] - (int)g->tokens[p][c];
            deficit += missing > 0 ? missing : 0;
        }
        shortfall[i] = deficit > gold ? deficit - gold : 0;
    }

    // Closest card: least short, then most points. Reserving needs a face-up one.
    int close = 0;
    int close_up = 0;
    int best_close = 1 << 30;
    int best_close_up = 1 << 30;
    for (int i = 0; i < NUM_BUYS; i++) {
        int points = card[i] < 0 ? 0 : CARDS[card[i]].points;
        int rank = 64*shortfall[i] - points;
        if (rank < best_close) {
            best_close = rank;
            close = i;
        }
        if (i < 12 && rank < best_close_up) {
            best_close_up = rank;
            close_up = i;
        }
    }
    const int* need_c = need[close];
    int short_c = shortfall[close];

    // Buy: most points, then higher tier, then cheaper.
    int buy = -1;
    int best_buy = -(1 << 30);
    for (int i = 0; i < NUM_BUYS; i++) {
        if (!mask[ACT_BUY_FACEUP + i]) {
            continue;
        }
        int cost_sum = 0;
        for (int c = 0; c < NUM_COLORS; c++) {
            cost_sum += CARDS[card[i]].cost[c];
        }
        int score = 10000*CARDS[card[i]].points + 100*(CARDS[card[i]].tier - 1) - cost_sum;
        if (score > best_buy) {
            best_buy = score;
            buy = ACT_BUY_FACEUP + i;
        }
    }
    if (buy >= 0) {
        return buy;
    }

    // Take: the legal take that most reduces the closest card's shortfall.
    int held = token_total(g, p);
    int take = -1;
    int first_take = -1;
    float best_take = -1e9f;
    int take_gain = 0;
    for (int a = 0; a < NUM_TAKES; a++) {
        if (!mask[a]) {
            continue;
        }
        if (first_take < 0) {
            first_take = a;
        }
        int delta[NUM_COLORS];
        take_delta(a, delta);
        int size = 0;
        int deficit = 0;
        for (int c = 0; c < NUM_COLORS; c++) {
            size += delta[c];
            int missing = need_c[c] - (int)g->tokens[p][c] - delta[c];
            deficit += missing > 0 ? missing : 0;
        }
        int new_short = deficit > gold ? deficit - gold : 0;
        int gain = short_c - new_short;
        float score = (float)gain + 0.01f*(float)size;
        int same = a - ACT_TAKE2SAME;
        if (a >= ACT_TAKE2SAME && need_c[same] - (int)g->tokens[p][same] >= 2) {
            score += 0.25f;  // a 2-of-a-kind take covers a hole of 2 or more
        }
        if (held + size > MAX_TOKENS) {
            score -= 100.0f;  // avoid walking into a discard
        }
        if (score > best_take) {
            best_take = score;
            take = a;
            take_gain = gain;
        }
    }
    if (take >= 0 && take_gain > 0) {
        return take;
    }
    if (mask[ACT_RESERVE_FACEUP + close_up]) {
        return ACT_RESERVE_FACEUP + close_up;
    }
    if (first_take >= 0) {
        return first_take;
    }

    if (g->phase == PHASE_DISCARD) {  // give back whatever the closest card needs least
        int drop = -1;
        float best_drop = -1e9f;
        for (int c = 0; c < 6; c++) {
            if (!mask[ACT_DISCARD + c]) {
                continue;
            }
            float score = -1e5f;  // gold is always the last thing to hand back
            if (c < NUM_COLORS) {
                score = (float)((int)g->tokens[p][c] - need_c[c]) + 0.01f*(float)g->tokens[p][c];
            }
            if (score > best_drop) {
                best_drop = score;
                drop = ACT_DISCARD + c;
            }
        }
        if (drop >= 0) {
            return drop;
        }
    }
    for (int a = 0; a < ACT_PASS; a++) {
        if (mask[a]) {
            return a;
        }
    }
    return ACT_PASS;
}

void puf_set_bot_policy(Splendor* env, int bot_policy) {
    env->bot_policy = bot_policy;
}

// Refreshes what the trainer reads: every seat's observation row and, when the
// trainer has handed out mask buffers, every seat's 72-byte legal mask.
static void write_agents(Splendor* env) {
    unsigned char* rows[SPLENDOR_PLAYERS];
    for (int s = 0; s < SPLENDOR_PLAYERS; s++) {
        rows[s] = env->agents[s].observations;
    }
    game_obs(&env->game, rows);
    for (int s = 0; s < SPLENDOR_PLAYERS; s++) {
        if (env->agents[s].action_mask != NULL) {
            game_seat_mask(&env->game, s, env->agents[s].action_mask);
        }
    }
}

void puf_reset(Splendor* env) {
    game_reset(&env->game);
    write_agents(env);
}

void puf_step(Splendor* env) {
    Game* g = &env->game;
    for (int s = 0; s < SPLENDOR_PLAYERS; s++) {
        env->agents[s].rewards[0] = 0.0f;
        env->agents[s].terminals[0] = 0.0f;
    }

    int seat = g->current;
    int action = (int)env->agents[seat].actions[0];
    if (env->bot_policy == 1 && seat > 0) {
        action = bot_random(env);
    } else if (env->bot_policy == 2 && seat > 0) {
        action = bot_greedy(env);
    }

    GameStep step;
    game_apply(g, action, &step);
    env->agents[seat].rewards[0] = env->reward_point*(float)step.points
        + env->reward_card*(float)step.cards;
    if (!step.done) {
        write_agents(env);
        return;
    }

    float sum_points = 0.0f;
    float sum_cards = 0.0f;
    float sum_nobles = 0.0f;
    int best_points = 0;
    for (int s = 0; s < SPLENDOR_PLAYERS; s++) {
        // A shared top spot is a draw: neither win nor loss for the tied seats.
        if (!step.won[s]) {
            env->agents[s].rewards[0] += env->reward_loss;
        } else {
            best_points = g->points[s];
            if (step.num_winners == 1) {
                env->agents[s].rewards[0] += env->reward_win;
            }
        }
        env->agents[s].terminals[0] = 1.0f;
        sum_points += g->points[s];
        sum_cards += g->num_bought[s];
        sum_nobles += g->num_nobles_p[s];
    }

    float seat0 = 0.0f;
    if (step.won[0]) {
        seat0 = (step.num_winners == 1) ? 1.0f : 0.5f;
    }
    env->log.perf += seat0;
    env->log.score += (float)best_points;
    env->log.points += sum_points / (float)SPLENDOR_PLAYERS;
    env->log.cards += sum_cards / (float)SPLENDOR_PLAYERS;
    env->log.nobles += sum_nobles / (float)SPLENDOR_PLAYERS;
    env->log.game_length += (float)g->turn;
    env->log.invalid += (float)g->invalid;
    env->log.n += 1.0f;

    puf_reset(env);  // 5.0 always auto-resets: these obs are the new game's
}

const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};
const Color PUFF_PANEL = (Color){12, 40, 40, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_DIM = (Color){110, 140, 140, 255};

// White, blue, green, red, black, gold.
const Color GEM_COLOR[6] = {
    (Color){238, 238, 230, 255},
    (Color){40, 110, 230, 255},
    (Color){40, 180, 110, 255},
    (Color){215, 60, 60, 255},
    (Color){60, 60, 72, 255},
    (Color){232, 190, 70, 255},
};

static void draw_gem(int x, int y, int radius, int color, int count) {
    DrawCircle(x, y, radius, GEM_COLOR[color]);
    DrawCircleLines(x, y, radius, PUFF_BACKGROUND);
    const char* text = TextFormat("%d", count);
    int w = MeasureText(text, 16);
    DrawText(text, x - w/2, y - 8, 16, color == 0 ? PUFF_BACKGROUND : PUFF_WHITE);
}

static void draw_dev_card(int x, int y, int w, int h, int card_id) {
    if (card_id < 0) {
        DrawRectangleLines(x, y, w, h, PUFF_PANEL);
        return;
    }
    const Card* c = &CARDS[card_id];
    DrawRectangle(x, y, w, h, PUFF_PANEL);
    DrawRectangle(x, y, w, 18, GEM_COLOR[c->bonus]);
    DrawRectangleLines(x, y, w, h, PUFF_DIM);
    if (c->points > 0) {
        DrawText(TextFormat("%d", c->points), x + w - 14, y + 1, 16,
            c->bonus == 0 ? PUFF_BACKGROUND : PUFF_WHITE);
    }
    int row = y + 24;
    for (int color = 0; color < NUM_COLORS; color++) {
        if (c->cost[color] == 0) {
            continue;
        }
        DrawCircle(x + 12, row + 8, 7, GEM_COLOR[color]);
        DrawText(TextFormat("%d", c->cost[color]), x + 24, row + 1, 15, PUFF_WHITE);
        row += 17;
    }
}

void puf_render(Splendor* env) {
    Game* g = &env->game;
    if (!IsWindowReady()) {
        InitWindow(1180, 760, "PufferLib Splendor");
        SetTargetFPS(60);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }
    static const char* PHASE_NAME[3] = {"to move", "must discard", "picks a noble"};

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);
    DrawText(TextFormat("Splendor  turn %d   seat %d %s", g->turn, g->current,
        PHASE_NAME[g->phase]), 20, 16, 22, PUFF_WHITE);

    DrawText("bank", 20, 54, 16, PUFF_DIM);
    for (int c = 0; c < 6; c++) {
        draw_gem(80 + 46*c, 62, 17, c, g->bank[c]);
    }

    DrawText("nobles", 400, 54, 16, PUFF_DIM);
    for (int i = 0; i < g->num_nobles; i++) {
        int x = 470 + 96*i;
        DrawRectangle(x, 44, 88, 38, PUFF_PANEL);
        DrawRectangleLines(x, 44, 88, 38, g->nobles[i] < 0 ? PUFF_PANEL : PUFF_CYAN);
        if (g->nobles[i] < 0) {
            DrawText("taken", x + 24, 56, 14, PUFF_DIM);
            continue;
        }
        int col = 0;
        for (int c = 0; c < NUM_COLORS; c++) {
            if (NOBLES[g->nobles[i]][c] == 0) {
                continue;
            }
            DrawCircle(x + 14 + 28*col, 63, 8, GEM_COLOR[c]);
            DrawText(TextFormat("%d", NOBLES[g->nobles[i]][c]), x + 24 + 28*col, 56, 14,
                PUFF_WHITE);
            col++;
        }
    }

    for (int t = 2; t >= 0; t--) {
        int y = 110 + 124*(2 - t);
        DrawText(TextFormat("tier %d", t + 1), 20, y + 40, 16, PUFF_DIM);
        DrawText(TextFormat("deck %d", g->deck_n[t]), 20, y + 62, 14, PUFF_DIM);
        for (int pos = 0; pos < 4; pos++) {
            draw_dev_card(90 + 108*pos, y, 96, 112, g->faceup[t][pos]);
        }
    }

    int panel_x = 540;
    DrawText("players", panel_x, 92, 16, PUFF_DIM);
    for (int p = 0; p < SPLENDOR_PLAYERS; p++) {
        int y = 114 + 124*p;
        DrawRectangle(panel_x, y, 610, 112, PUFF_PANEL);
        DrawRectangleLines(panel_x, y, 610, 112, p == g->current ? PUFF_CYAN : PUFF_PANEL);
        DrawText(TextFormat("seat %d%s   %d points   %d cards   %d nobles", p,
            p == 0 ? " (learner)" : "", g->points[p], g->num_bought[p],
            g->num_nobles_p[p]), panel_x + 12, y + 8, 16,
            p == g->current ? PUFF_CYAN : PUFF_WHITE);
        DrawText("tokens", panel_x + 12, y + 36, 14, PUFF_DIM);
        for (int c = 0; c < 6; c++) {
            draw_gem(panel_x + 80 + 40*c, y + 44, 15, c, g->tokens[p][c]);
        }
        DrawText("bonus", panel_x + 12, y + 74, 14, PUFF_DIM);
        for (int c = 0; c < NUM_COLORS; c++) {
            draw_gem(panel_x + 80 + 40*c, y + 82, 15, c, g->bonuses[p][c]);
        }
        for (int r = 0; r < MAX_RESERVED; r++) {
            int x = panel_x + 340 + 90*r;
            if (g->reserved[p][r] < 0) {
                DrawRectangleLines(x, y + 34, 82, 70, PUFF_BACKGROUND);
                continue;
            }
            if (g->hidden[p][r] && p != 0) {
                DrawRectangle(x, y + 34, 82, 70, PUFF_BACKGROUND);
                DrawText("hidden", x + 18, y + 62, 14, PUFF_DIM);
                continue;
            }
            draw_dev_card(x, y + 34, 82, 70, g->reserved[p][r]);
        }
    }
    EndDrawing();
    puf_web_vsync();
}

void puf_close(Splendor* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
}

void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = SPLENDOR_PLAYERS;
    env->game.num_players = SPLENDOR_PLAYERS;
    env->game.max_turns = (int)dict_get(kwargs, "max_turns");
    if (env->game.max_turns <= 0) {
        env->game.max_turns = 60*SPLENDOR_PLAYERS;  // 60 turns per seat
    }
    env->reward_point = dict_get(kwargs, "reward_point");
    env->reward_card = dict_get(kwargs, "reward_card");
    env->reward_win = dict_get(kwargs, "reward_win");
    env->reward_loss = dict_get(kwargs, "reward_loss");
    env->bot_policy = (int)dict_get(kwargs, "bot_policy");
    // env->rng is the env index (pufferl env_setup sets it before puf_init);
    // splitmix it so neighbouring envs do not deal correlated games.
    env->game.rng = seed_rng(env->rng);

    // Seat 0 is the learner; the rest play the historical policy on the
    // hist_policy_percent tail of envs (the trainer forces every seat to
    // policy 0 on the selfplay majority). Mask buffers arrive after init.
    for (int s = 0; s < SPLENDOR_PLAYERS; s++) {
        env->agents[s].policy = (s == 0) ? 0 : 1;
        env->agents[s].action_mask = NULL;
    }
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "points", log->points);
    dict_set(out, "cards", log->cards);
    dict_set(out, "nobles", log->nobles);
    dict_set(out, "game_length", log->game_length);
    dict_set(out, "invalid", log->invalid);
    dict_set(out, "n", log->n);
}
