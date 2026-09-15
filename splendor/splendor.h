// PufferLib 3.0 front-end for the Splendor game core. One Splendor struct =
// one game with num_players seats; one PufferLib agent per seat; one c_step =
// one turn by the seat to move. All rules, tables, observations and masks live
// in game.h; this file is only buffer plumbing, rewards and logging.
#ifndef SPLENDOR_H
#define SPLENDOR_H

#include "game.h"

typedef struct Log Log;
struct Log {
    float score;        // winner's points
    float points;       // mean final points over seats
    float cards;        // mean cards bought per seat
    float nobles;       // mean nobles per seat
    float game_length;  // turns played
    float perf;         // 1.0 if the game ended by someone reaching 15 points
    float invalid;      // illegal actions submitted during the game
    float n;            // MUST be last (env_binding.h iterates Log as floats)
};

typedef struct Splendor Splendor;
struct Splendor {
    Log log;                      // must be first
    unsigned char* observations;  // num_players rows of GAME_OBS_SIZE(num_players)
    int* actions;                 // num_players ints
    float* rewards;               // num_players floats
    unsigned char* terminals;     // num_players bytes

    float reward_point;
    float reward_card;
    float reward_win;
    float reward_loss;
    int auto_reset;               // 1 (training): deal a new game at game end

    Game game;                    // the whole snapshot region (see binding.c)
};

// Every seat's observation row of the one contiguous buffer PufferLib gave us.
static void write_obs(Splendor* env) {
    int P = env->game.num_players;
    int obs_n = GAME_OBS_SIZE(P);
    unsigned char* rows[MAX_PLAYERS];
    for (int me = 0; me < P; me++) {
        rows[me] = env->observations + obs_n*me;
    }
    game_obs(&env->game, rows);
}

void c_reset(Splendor* env) {
    game_reset(&env->game);
    write_obs(env);
}

void c_step(Splendor* env) {
    Game* g = &env->game;
    int P = g->num_players;
    for (int p = 0; p < P; p++) {
        env->rewards[p] = 0.0f;
        env->terminals[p] = 0;
    }

    int seat = g->current;
    int action = env->actions[seat];
    if (action == ACT_NOOP || g->game_over) {
        return;  // frozen (waiting for another agent) or finished and held
    }

    GameStep step;
    game_apply(g, action, &step);
    env->rewards[seat] = env->reward_point*(float)step.points
        + env->reward_card*(float)step.cards;
    if (!step.done) {
        write_obs(env);
        return;
    }

    float sum_points = 0.0f;
    float sum_cards = 0.0f;
    float sum_nobles = 0.0f;
    int best_points = 0;
    for (int p = 0; p < P; p++) {
        // A shared top spot is a draw: 0 for the tied players.
        if (!step.won[p]) {
            env->rewards[p] += env->reward_loss;
        } else if (step.num_winners == 1) {
            env->rewards[p] += env->reward_win;
        }
        if (step.won[p]) {
            best_points = g->points[p];
        }
        env->terminals[p] = 1;
        sum_points += g->points[p];
        sum_cards += g->num_bought[p];
        sum_nobles += g->num_nobles_p[p];
    }

    env->log.score += (float)best_points;
    env->log.points += sum_points / (float)P;
    env->log.cards += sum_cards / (float)P;
    env->log.nobles += sum_nobles / (float)P;
    env->log.game_length += (float)g->turn;
    env->log.perf += step.by_points ? 1.0f : 0.0f;
    env->log.invalid += (float)g->invalid;
    env->log.n += 1.0f;

    if (env->auto_reset) {
        c_reset(env);  // the obs returned by this step belong to the new game
        return;
    }
    g->game_over = 1;  // keep the final board until an explicit reset
    game_mask(g);
    write_obs(env);
}

void c_render(Splendor* env) {
    game_render_text(&env->game);
}

void c_close(Splendor* env) {
    (void)env;  // the env owns no heap memory
}

#endif  // SPLENDOR_H
