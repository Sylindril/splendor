// Splendor game core: rules, tables, observations and legal-action masks.
// Pure C, no PufferLib types and no I/O outside game_render_text. Both the
// PufferLib 3.0 binding (splendor/splendor.h) and the PufferLib 5.0 front-end
// (pufferlib5/splendor.h) include this file and add only buffer plumbing.
// See DESIGN.md for the rules, the action table and the observation layout.
#ifndef SPLENDOR_GAME_H
#define SPLENDOR_GAME_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_PLAYERS 4
#define NUM_COLORS 5
#define GEM_GOLD 5
#define NUM_CARDS 90
#define NUM_NOBLES 10
#define NOBLE_MAX 5
#define CARD_N 11
#define PLAYER_N 48
#define NUM_ACTIONS 72
#define MAX_RESERVED 3
#define MAX_TOKENS 10
#define WIN_POINTS 15

// Action table (see DESIGN.md).
#define ACT_TAKE3 0
#define ACT_TAKE2 10
#define ACT_TAKE1 20
#define ACT_TAKE2SAME 25
#define ACT_RESERVE_FACEUP 30
#define ACT_RESERVE_DECK 42
#define ACT_BUY_FACEUP 45
#define ACT_BUY_RESERVED 57
#define ACT_PASS 60
#define ACT_DISCARD 61  // 6: return one token of color c (incl. gold) while over 10
#define ACT_NOBLE 67    // 5: choose dealt noble slot i when several qualify

#define ACT_NOOP -1     // action value meaning "do not step this game at all"

// Sub-phases within a turn (same seat keeps acting until back in PHASE_ACT).
#define PHASE_ACT 0
#define PHASE_DISCARD 1
#define PHASE_NOBLE 2

// Observation layout.
#define OBS_BANK 0
#define OBS_DECKS 6
#define OBS_FACEUP 9
#define OBS_NOBLES 141
#define OBS_PLAYERS 166
#define GAME_OBS_SIZE(P) (OBS_PLAYERS + PLAYER_N*(P) + NUM_ACTIONS + 2)  // ... + turn + to_move
#define OBS_MASK(P) (OBS_PLAYERS + PLAYER_N*(P))
#define OBS_TURN(P) (OBS_MASK(P) + NUM_ACTIONS)
#define OBS_MAX GAME_OBS_SIZE(MAX_PLAYERS)

typedef struct Card Card;
struct Card {
    unsigned char tier;      // 1..3
    unsigned char bonus;     // 0..4
    unsigned char points;
    unsigned char cost[NUM_COLORS];
};

// The 90 standard Splendor development cards, grouped by tier then bonus color.
// {tier, bonus, points, {white, blue, green, red, black}}
static const Card CARDS[NUM_CARDS] = {
    // Tier 1
    {1, 0, 0, {0, 0, 0, 2, 1}},  //  0 white
    {1, 0, 0, {0, 1, 1, 1, 1}},  //  1 white
    {1, 0, 0, {0, 1, 2, 1, 1}},  //  2 white
    {1, 0, 0, {0, 2, 0, 0, 2}},  //  3 white
    {1, 0, 0, {0, 2, 2, 0, 1}},  //  4 white
    {1, 0, 0, {0, 3, 0, 0, 0}},  //  5 white
    {1, 0, 0, {3, 1, 0, 0, 1}},  //  6 white
    {1, 0, 1, {0, 0, 4, 0, 0}},  //  7 white
    {1, 1, 0, {0, 0, 0, 0, 3}},  //  8 blue
    {1, 1, 0, {0, 0, 2, 0, 2}},  //  9 blue
    {1, 1, 0, {0, 1, 3, 1, 0}},  // 10 blue
    {1, 1, 0, {1, 0, 0, 0, 2}},  // 11 blue
    {1, 1, 0, {1, 0, 1, 1, 1}},  // 12 blue
    {1, 1, 0, {1, 0, 1, 2, 1}},  // 13 blue
    {1, 1, 0, {1, 0, 2, 2, 0}},  // 14 blue
    {1, 1, 1, {0, 0, 0, 4, 0}},  // 15 blue
    {1, 2, 0, {0, 0, 0, 3, 0}},  // 16 green
    {1, 2, 0, {0, 1, 0, 2, 2}},  // 17 green
    {1, 2, 0, {0, 2, 0, 2, 0}},  // 18 green
    {1, 2, 0, {1, 1, 0, 1, 1}},  // 19 green
    {1, 2, 0, {1, 1, 0, 1, 2}},  // 20 green
    {1, 2, 0, {1, 3, 1, 0, 0}},  // 21 green
    {1, 2, 0, {2, 1, 0, 0, 0}},  // 22 green
    {1, 2, 1, {0, 0, 0, 0, 4}},  // 23 green
    {1, 3, 0, {0, 2, 1, 0, 0}},  // 24 red
    {1, 3, 0, {1, 0, 0, 1, 3}},  // 25 red
    {1, 3, 0, {1, 1, 1, 0, 1}},  // 26 red
    {1, 3, 0, {2, 0, 0, 2, 0}},  // 27 red
    {1, 3, 0, {2, 0, 1, 0, 2}},  // 28 red
    {1, 3, 0, {2, 1, 1, 0, 1}},  // 29 red
    {1, 3, 0, {3, 0, 0, 0, 0}},  // 30 red
    {1, 3, 1, {4, 0, 0, 0, 0}},  // 31 red
    {1, 4, 0, {0, 0, 1, 3, 1}},  // 32 black
    {1, 4, 0, {0, 0, 2, 1, 0}},  // 33 black
    {1, 4, 0, {0, 0, 3, 0, 0}},  // 34 black
    {1, 4, 0, {1, 1, 1, 1, 0}},  // 35 black
    {1, 4, 0, {1, 2, 1, 1, 0}},  // 36 black
    {1, 4, 0, {2, 0, 2, 0, 0}},  // 37 black
    {1, 4, 0, {2, 2, 0, 1, 0}},  // 38 black
    {1, 4, 1, {0, 4, 0, 0, 0}},  // 39 black
    // Tier 2
    {2, 0, 1, {0, 0, 3, 2, 2}},  // 40 white
    {2, 0, 1, {2, 3, 0, 3, 0}},  // 41 white
    {2, 0, 2, {0, 0, 0, 5, 0}},  // 42 white
    {2, 0, 2, {0, 0, 0, 5, 3}},  // 43 white
    {2, 0, 2, {0, 0, 1, 4, 2}},  // 44 white
    {2, 0, 3, {6, 0, 0, 0, 0}},  // 45 white
    {2, 1, 1, {0, 2, 2, 3, 0}},  // 46 blue
    {2, 1, 1, {0, 2, 3, 0, 3}},  // 47 blue
    {2, 1, 2, {0, 5, 0, 0, 0}},  // 48 blue
    {2, 1, 2, {2, 0, 0, 1, 4}},  // 49 blue
    {2, 1, 2, {5, 3, 0, 0, 0}},  // 50 blue
    {2, 1, 3, {0, 6, 0, 0, 0}},  // 51 blue
    {2, 2, 1, {2, 3, 0, 0, 2}},  // 52 green
    {2, 2, 1, {3, 0, 2, 3, 0}},  // 53 green
    {2, 2, 2, {0, 0, 5, 0, 0}},  // 54 green
    {2, 2, 2, {0, 5, 3, 0, 0}},  // 55 green
    {2, 2, 2, {4, 2, 0, 0, 1}},  // 56 green
    {2, 2, 3, {0, 0, 6, 0, 0}},  // 57 green
    {2, 3, 1, {0, 3, 0, 2, 3}},  // 58 red
    {2, 3, 1, {2, 0, 0, 2, 3}},  // 59 red
    {2, 3, 2, {0, 0, 0, 0, 5}},  // 60 red
    {2, 3, 2, {1, 4, 2, 0, 0}},  // 61 red
    {2, 3, 2, {3, 0, 0, 0, 5}},  // 62 red
    {2, 3, 3, {0, 0, 0, 6, 0}},  // 63 red
    {2, 4, 1, {3, 0, 3, 0, 2}},  // 64 black
    {2, 4, 1, {3, 2, 2, 0, 0}},  // 65 black
    {2, 4, 2, {0, 0, 5, 3, 0}},  // 66 black
    {2, 4, 2, {0, 1, 4, 2, 0}},  // 67 black
    {2, 4, 2, {5, 0, 0, 0, 0}},  // 68 black
    {2, 4, 3, {0, 0, 0, 0, 6}},  // 69 black
    // Tier 3
    {3, 0, 3, {0, 3, 3, 5, 3}},  // 70 white
    {3, 0, 4, {0, 0, 0, 0, 7}},  // 71 white
    {3, 0, 4, {3, 0, 0, 3, 6}},  // 72 white
    {3, 0, 5, {3, 0, 0, 0, 7}},  // 73 white
    {3, 1, 3, {3, 0, 3, 3, 5}},  // 74 blue
    {3, 1, 4, {6, 3, 0, 0, 3}},  // 75 blue
    {3, 1, 4, {7, 0, 0, 0, 0}},  // 76 blue
    {3, 1, 5, {7, 3, 0, 0, 0}},  // 77 blue
    {3, 2, 3, {5, 3, 0, 3, 3}},  // 78 green
    {3, 2, 4, {0, 7, 0, 0, 0}},  // 79 green
    {3, 2, 4, {3, 6, 3, 0, 0}},  // 80 green
    {3, 2, 5, {0, 7, 3, 0, 0}},  // 81 green
    {3, 3, 3, {3, 5, 3, 0, 3}},  // 82 red
    {3, 3, 4, {0, 0, 7, 0, 0}},  // 83 red
    {3, 3, 4, {0, 3, 6, 3, 0}},  // 84 red
    {3, 3, 5, {0, 0, 7, 3, 0}},  // 85 red
    {3, 4, 3, {3, 3, 5, 3, 0}},  // 86 black
    {3, 4, 4, {0, 0, 0, 7, 0}},  // 87 black
    {3, 4, 4, {0, 0, 3, 6, 3}},  // 88 black
    {3, 4, 5, {0, 0, 0, 7, 3}},  // 89 black
};

// The 10 standard nobles (3 points each); value = required bonus cards.
static const unsigned char NOBLES[NUM_NOBLES][5] = {
    {4, 4, 0, 0, 0},
    {0, 4, 4, 0, 0},
    {0, 0, 4, 4, 0},
    {0, 0, 0, 4, 4},
    {4, 0, 0, 0, 4},
    {3, 3, 3, 0, 0},
    {0, 3, 3, 3, 0},
    {0, 0, 3, 3, 3},
    {3, 0, 0, 3, 3},
    {3, 3, 0, 0, 3},
};

// Take-3 / take-2 color combos, lexicographic (must match DESIGN.md action table).
static const unsigned char COMBOS3[10][3] = {
    {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4},
    {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
};
static const unsigned char COMBOS2[10][2] = {
    {0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2},
    {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4},
};

// Number of cards in each tier's deck and the id of its first card.
static const int TIER_SIZE[3] = {40, 30, 20};
static const int TIER_FIRST[3] = {0, 40, 70};

// One game. This is the whole mutable state: the front-ends add only buffer
// pointers, reward weights and logging, so a byte copy of a Game is a complete
// snapshot (splendor/binding.c hands those to Python as opaque bytes).
typedef struct Game Game;
struct Game {
    int num_players;
    int max_turns;  // total turns over all seats; the game ends when reached
    uint64_t rng;   // per-game RNG; we never touch rand()

    // Board
    unsigned char bank[6];            // 5 colors + gold
    unsigned char deck[3][40];        // card ids, shuffled; top of deck is the last entry
    int deck_n[3];
    short faceup[3][4];               // card id or -1
    short nobles[NOBLE_MAX];          // noble id or -1 (taken); only num_nobles are dealt
    int num_nobles;

    // Players
    unsigned char tokens[MAX_PLAYERS][6];
    unsigned char bonuses[MAX_PLAYERS][NUM_COLORS];
    unsigned char points[MAX_PLAYERS];
    unsigned char num_bought[MAX_PLAYERS];
    unsigned char num_nobles_p[MAX_PLAYERS];
    short reserved[MAX_PLAYERS][MAX_RESERVED];       // card id or -1
    unsigned char hidden[MAX_PLAYERS][MAX_RESERVED]; // 1 = reserved from a deck top

    int start_seat;
    int current;
    int turn;
    int invalid;                      // illegal actions submitted this game
    int phase;                        // PHASE_ACT / PHASE_DISCARD / PHASE_NOBLE
    int game_over;                    // held final board (3.0 auto_reset = 0 only)
    unsigned char mask[NUM_ACTIONS];  // cached legal mask of the current seat
};

// What one game_apply did, so a front-end can pay rewards and write a log
// without knowing the rules.
typedef struct GameStep GameStep;
struct GameStep {
    int seat;        // the seat that acted
    int points;      // points it gained this step
    int cards;       // cards it bought this step
    int invalid;     // 1 if the submitted action was illegal
    int done;        // 1 if the game ended on this step
    int by_points;   // done because someone reached WIN_POINTS (else max_turns)
    int num_winners; // > 1 means a draw among them
    unsigned char won[MAX_PLAYERS];
};

// xorshift64* -- fast, deterministic, per-game.
static inline uint64_t rnd(Game* g) {
    uint64_t x = g->rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g->rng = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline int rnd_int(Game* g, int n) {
    return (int)(rnd(g) % (uint64_t)n);
}

// splitmix64: spreads a small seed (0, 1, 2, ...) over the whole 64-bit state.
static inline uint64_t seed_rng(uint64_t seed) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return z ? z : 0x9E3779B97F4A7C15ULL;  // xorshift must not start at 0
}

static inline int token_total(Game* g, int p) {
    const unsigned char* t = g->tokens[p];
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5];
}

static inline int num_reserved(Game* g, int p) {
    int n = 0;
    for (int r = 0; r < MAX_RESERVED; r++) {
        n += (g->reserved[p][r] >= 0);
    }
    return n;
}

// Affordable = the gold on hand covers every shortfall after bonuses and tokens.
static inline int affordable(Game* g, int p, int card_id) {
    if (card_id < 0) {
        return 0;
    }
    const unsigned char* cost = CARDS[card_id].cost;
    int shortfall = 0;
    for (int c = 0; c < NUM_COLORS; c++) {
        int need = (int)cost[c] - (int)g->bonuses[p][c];
        if (need <= 0) {
            continue;
        }
        need -= (int)g->tokens[p][c];
        if (need > 0) {
            shortfall += need;
        }
    }
    return shortfall <= (int)g->tokens[p][GEM_GOLD];
}

static inline int noble_qualifies(Game* g, int p, int id) {
    for (int c = 0; c < NUM_COLORS; c++) {
        if (g->bonuses[p][c] < NOBLES[id][c]) {
            return 0;
        }
    }
    return 1;
}

// Legal mask for the seat about to act. An action is set iff applying it is valid.
static void game_mask(Game* g) {
    int p = g->current;
    unsigned char* mask = g->mask;
    memset(mask, 0, NUM_ACTIONS);
    if (g->game_over) {
        return;  // nothing is legal on a finished board
    }

    if (g->phase == PHASE_DISCARD) {  // over 10 tokens: must return one
        for (int c = 0; c < 6; c++) {
            mask[ACT_DISCARD + c] = g->tokens[p][c] > 0;
        }
        return;
    }
    if (g->phase == PHASE_NOBLE) {  // several nobles qualify: choose one
        for (int i = 0; i < g->num_nobles; i++) {
            int id = g->nobles[i];
            mask[ACT_NOBLE + i] = (id >= 0) && noble_qualifies(g, p, id);
        }
        return;
    }

    int any = 0;
    int reserved = num_reserved(g, p);
    // Takes are limited only by the bank; going over 10 tokens triggers a discard phase.
    for (int i = 0; i < 10; i++) {
        const unsigned char* c = COMBOS3[i];
        if (g->bank[c[0]] && g->bank[c[1]] && g->bank[c[2]]) {
            mask[ACT_TAKE3 + i] = 1;
            any = 1;
        }
    }
    for (int i = 0; i < 10; i++) {
        const unsigned char* c = COMBOS2[i];
        if (g->bank[c[0]] && g->bank[c[1]]) {
            mask[ACT_TAKE2 + i] = 1;
            any = 1;
        }
    }
    for (int c = 0; c < NUM_COLORS; c++) {
        if (g->bank[c] >= 4) {
            mask[ACT_TAKE2SAME + c] = 1;
            any = 1;
        }
        if (g->bank[c]) {
            mask[ACT_TAKE1 + c] = 1;
            any = 1;
        }
    }
    if (reserved < MAX_RESERVED) {
        for (int t = 0; t < 3; t++) {
            for (int pos = 0; pos < 4; pos++) {
                if (g->faceup[t][pos] >= 0) {
                    mask[ACT_RESERVE_FACEUP + 4*t + pos] = 1;
                    any = 1;
                }
            }
            if (g->deck_n[t] > 0) {
                mask[ACT_RESERVE_DECK + t] = 1;
                any = 1;
            }
        }
    }
    for (int t = 0; t < 3; t++) {
        for (int pos = 0; pos < 4; pos++) {
            if (affordable(g, p, g->faceup[t][pos])) {
                mask[ACT_BUY_FACEUP + 4*t + pos] = 1;
                any = 1;
            }
        }
    }
    for (int r = 0; r < MAX_RESERVED; r++) {
        if (affordable(g, p, g->reserved[p][r])) {
            mask[ACT_BUY_RESERVED + r] = 1;
            any = 1;
        }
    }
    // Pass is legal only when nothing else is.
    mask[ACT_PASS] = !any;
}

// The 72-byte mask one seat sees: the cached mask if it is to move, else
// PASS only (a masked policy then has zero gradient on an idle seat).
static void game_seat_mask(Game* g, int seat, unsigned char* out) {
    if (seat == g->current) {
        memcpy(out, g->mask, NUM_ACTIONS);
        return;
    }
    memset(out, 0, NUM_ACTIONS);
    out[ACT_PASS] = !g->game_over;
}

static inline void write_card(unsigned char* obs, int card_id) {
    if (card_id < 0) {
        memset(obs, 0, CARD_N);
        return;
    }
    const Card* card = &CARDS[card_id];
    for (int c = 0; c < NUM_COLORS; c++) {
        obs[c] = card->cost[c];
        obs[NUM_COLORS + c] = 0;
    }
    obs[NUM_COLORS + card->bonus] = 1;
    obs[2*NUM_COLORS] = card->points;
}

// Writes every seat's observation into rows[0 .. num_players-1]. Seats see
// themselves first, then the other players in turn order; opponents' deck-top
// reserves are hidden.
static void game_obs(Game* g, unsigned char** rows) {
    int P = g->num_players;

    // Everything before the player blocks is public and identical for all seats.
    unsigned char shared[OBS_PLAYERS];
    memset(shared, 0, sizeof(shared));
    memcpy(shared + OBS_BANK, g->bank, 6);
    for (int t = 0; t < 3; t++) {
        shared[OBS_DECKS + t] = (unsigned char)g->deck_n[t];
        for (int pos = 0; pos < 4; pos++) {
            write_card(shared + OBS_FACEUP + CARD_N*(4*t + pos), g->faceup[t][pos]);
        }
    }
    for (int i = 0; i < g->num_nobles; i++) {
        int id = g->nobles[i];
        if (id >= 0) {
            memcpy(shared + OBS_NOBLES + NUM_COLORS*i, NOBLES[id], NUM_COLORS);
        }
    }

    // Two 48-byte blocks per player: one for its owner, one for everyone else.
    unsigned char own[MAX_PLAYERS][PLAYER_N];
    unsigned char pub[MAX_PLAYERS][PLAYER_N];
    for (int p = 0; p < P; p++) {
        unsigned char* o = own[p];
        memset(o, 0, PLAYER_N);
        memcpy(o, g->tokens[p], 6);
        memcpy(o + 6, g->bonuses[p], NUM_COLORS);
        o[11] = g->points[p];
        for (int r = 0; r < MAX_RESERVED; r++) {
            int id = g->reserved[p][r];
            unsigned char* slot = o + 12 + 12*r;
            write_card(slot, id);
            slot[CARD_N] = (id >= 0);
        }
        memcpy(pub[p], o, PLAYER_N);
        for (int r = 0; r < MAX_RESERVED; r++) {
            if (g->reserved[p][r] >= 0 && g->hidden[p][r]) {
                // Opponents only learn that a hidden card exists.
                memset(pub[p] + 12 + 12*r, 0, CARD_N);
            }
        }
    }

    unsigned char turn = g->turn > 255 ? 255 : (unsigned char)g->turn;
    for (int me = 0; me < P; me++) {
        unsigned char* obs = rows[me];
        memcpy(obs, shared, OBS_PLAYERS);
        for (int j = 0; j < P; j++) {
            int p = (me + j) % P;
            memcpy(obs + OBS_PLAYERS + PLAYER_N*j, j == 0 ? own[p] : pub[p], PLAYER_N);
        }
        game_seat_mask(g, me, obs + OBS_MASK(P));
        obs[OBS_TURN(P)] = turn;
        obs[OBS_TURN(P) + 1] = (me == g->current);  // 1 iff it is this seat's move
    }
}

static inline int draw_card(Game* g, int tier) {
    if (g->deck_n[tier] <= 0) {
        return -1;
    }
    g->deck_n[tier]--;
    return g->deck[tier][g->deck_n[tier]];
}

static inline void take_noble(Game* g, int p, int i) {
    g->nobles[i] = -1;
    g->points[p] += 3;
    g->num_nobles_p[p]++;
}

// Nobles visit at the end of the turn, at most one per turn. Exactly one
// qualifying noble is taken automatically; with several, the player chooses.
static void check_nobles(Game* g, int p) {
    int count = 0;
    int first = -1;
    for (int i = 0; i < g->num_nobles; i++) {
        int id = g->nobles[i];
        if (id >= 0 && noble_qualifies(g, p, id)) {
            count++;
            if (first < 0) {
                first = i;
            }
        }
    }
    if (count == 1) {
        take_noble(g, p, first);
    } else if (count > 1) {
        g->phase = PHASE_NOBLE;
    }
}

// Pay tokens first, gold only for the shortfall. Spent tokens go back to the bank.
static void pay_for(Game* g, int p, int card_id) {
    const unsigned char* cost = CARDS[card_id].cost;
    int gold = 0;
    for (int c = 0; c < NUM_COLORS; c++) {
        int need = (int)cost[c] - (int)g->bonuses[p][c];
        if (need <= 0) {
            continue;
        }
        int pay = need < g->tokens[p][c] ? need : g->tokens[p][c];
        g->tokens[p][c] -= pay;
        g->bank[c] += pay;
        gold += need - pay;
    }
    g->tokens[p][GEM_GOLD] -= gold;
    g->bank[GEM_GOLD] += gold;
}

static void buy_card(Game* g, int p, int card_id) {
    pay_for(g, p, card_id);
    const Card* card = &CARDS[card_id];
    g->bonuses[p][card->bonus]++;
    g->points[p] += card->points;
    g->num_bought[p]++;
    check_nobles(g, p);
}

static void reserve_card(Game* g, int p, int card_id, int is_hidden) {
    for (int r = 0; r < MAX_RESERVED; r++) {
        if (g->reserved[p][r] < 0) {
            g->reserved[p][r] = (short)card_id;
            g->hidden[p][r] = (unsigned char)is_hidden;
            break;
        }
    }
    // Reserving always grants a gold if the bank has one (discard later if over 10).
    if (g->bank[GEM_GOLD] > 0) {
        g->bank[GEM_GOLD]--;
        g->tokens[p][GEM_GOLD]++;
    }
}

// Applies an action already known to be legal (or PASS).
static void apply_action(Game* g, int p, int action) {
    if (action < ACT_TAKE2) {
        const unsigned char* c = COMBOS3[action - ACT_TAKE3];
        for (int i = 0; i < 3; i++) {
            g->bank[c[i]]--;
            g->tokens[p][c[i]]++;
        }
    } else if (action < ACT_TAKE1) {
        const unsigned char* c = COMBOS2[action - ACT_TAKE2];
        for (int i = 0; i < 2; i++) {
            g->bank[c[i]]--;
            g->tokens[p][c[i]]++;
        }
    } else if (action < ACT_TAKE2SAME) {
        int c = action - ACT_TAKE1;
        g->bank[c]--;
        g->tokens[p][c]++;
    } else if (action < ACT_RESERVE_FACEUP) {
        int c = action - ACT_TAKE2SAME;
        g->bank[c] -= 2;
        g->tokens[p][c] += 2;
    } else if (action < ACT_RESERVE_DECK) {
        int slot = action - ACT_RESERVE_FACEUP;
        int tier = slot / 4;
        int pos = slot % 4;
        int card_id = g->faceup[tier][pos];
        g->faceup[tier][pos] = (short)draw_card(g, tier);
        reserve_card(g, p, card_id, 0);
    } else if (action < ACT_BUY_FACEUP) {
        int tier = action - ACT_RESERVE_DECK;
        reserve_card(g, p, draw_card(g, tier), 1);
    } else if (action < ACT_BUY_RESERVED) {
        int slot = action - ACT_BUY_FACEUP;
        int tier = slot / 4;
        int pos = slot % 4;
        int card_id = g->faceup[tier][pos];
        g->faceup[tier][pos] = (short)draw_card(g, tier);
        buy_card(g, p, card_id);
    } else if (action < ACT_PASS) {
        int r = action - ACT_BUY_RESERVED;
        int card_id = g->reserved[p][r];
        g->reserved[p][r] = -1;
        g->hidden[p][r] = 0;
        buy_card(g, p, card_id);
    } else if (action == ACT_PASS) {
        // nothing happens
    } else if (action < ACT_NOBLE) {
        int c = action - ACT_DISCARD;  // return one token to the bank
        g->tokens[p][c]--;
        g->bank[c]++;
    } else {
        take_noble(g, p, action - ACT_NOBLE);
    }
}

// Re-deals everything `seat` cannot see (deck order and opponents' deck-top
// reserves) uniformly at random, tier by tier. Used to determinize a hidden-
// information state before searching it.
void game_determinize(Game* g, int seat) {
    for (int t = 0; t < 3; t++) {
        unsigned char pool[40];
        int n = 0;
        for (int i = 0; i < g->deck_n[t]; i++) {
            pool[n++] = g->deck[t][i];
        }
        int hidden_slots[MAX_PLAYERS * MAX_RESERVED][2];
        int h = 0;
        for (int p = 0; p < g->num_players; p++) {
            if (p == seat) {
                continue;
            }
            for (int r = 0; r < MAX_RESERVED; r++) {
                int id = g->reserved[p][r];
                if (id >= 0 && g->hidden[p][r] && CARDS[id].tier == t + 1) {
                    pool[n++] = (unsigned char)id;
                    hidden_slots[h][0] = p;
                    hidden_slots[h][1] = r;
                    h++;
                }
            }
        }
        for (int i = n - 1; i > 0; i--) {
            int j = rnd_int(g, i + 1);
            unsigned char tmp = pool[i];
            pool[i] = pool[j];
            pool[j] = tmp;
        }
        for (int i = 0; i < h; i++) {
            g->reserved[hidden_slots[i][0]][hidden_slots[i][1]] = pool[i];
        }
        for (int i = h; i < n; i++) {
            g->deck[t][i - h] = pool[i];
        }
    }
}

// Deals a fresh game and leaves the mask ready for the first seat to move.
// The caller writes the observations.
static void game_reset(Game* g) {
    int P = g->num_players;
    int per_color = (P == 2) ? 4 : (P == 3) ? 5 : 7;
    for (int c = 0; c < NUM_COLORS; c++) {
        g->bank[c] = (unsigned char)per_color;
    }
    g->bank[GEM_GOLD] = 5;

    for (int t = 0; t < 3; t++) {
        int n = TIER_SIZE[t];
        for (int i = 0; i < n; i++) {
            g->deck[t][i] = (unsigned char)(TIER_FIRST[t] + i);
        }
        for (int i = n - 1; i > 0; i--) {  // Fisher-Yates
            int j = rnd_int(g, i + 1);
            unsigned char tmp = g->deck[t][i];
            g->deck[t][i] = g->deck[t][j];
            g->deck[t][j] = tmp;
        }
        g->deck_n[t] = n;
        for (int pos = 0; pos < 4; pos++) {
            g->faceup[t][pos] = (short)draw_card(g, t);
        }
    }

    unsigned char noble_ids[NUM_NOBLES];
    for (int i = 0; i < NUM_NOBLES; i++) {
        noble_ids[i] = (unsigned char)i;
    }
    for (int i = NUM_NOBLES - 1; i > 0; i--) {
        int j = rnd_int(g, i + 1);
        unsigned char tmp = noble_ids[i];
        noble_ids[i] = noble_ids[j];
        noble_ids[j] = tmp;
    }
    g->num_nobles = P + 1;
    for (int i = 0; i < NOBLE_MAX; i++) {
        g->nobles[i] = (i < g->num_nobles) ? noble_ids[i] : -1;
    }

    memset(g->tokens, 0, sizeof(g->tokens));
    memset(g->bonuses, 0, sizeof(g->bonuses));
    memset(g->points, 0, sizeof(g->points));
    memset(g->num_bought, 0, sizeof(g->num_bought));
    memset(g->num_nobles_p, 0, sizeof(g->num_nobles_p));
    memset(g->hidden, 0, sizeof(g->hidden));
    for (int p = 0; p < MAX_PLAYERS; p++) {
        for (int r = 0; r < MAX_RESERVED; r++) {
            g->reserved[p][r] = -1;
        }
    }

    g->start_seat = rnd_int(g, P);  // so seat index carries no advantage
    g->current = g->start_seat;
    g->turn = 0;
    g->invalid = 0;
    g->phase = PHASE_ACT;
    g->game_over = 0;
    game_mask(g);
}

// Plays one action for the seat to move: a whole turn, or one step of a
// discard / noble sub-phase. Illegal actions are played as a pass (or, inside
// a sub-phase, as the first legal option) and counted. On return the mask is
// up to date unless the game ended, which the caller handles: it scores
// `out`, then either deals a new game or holds the final board.
static void game_apply(Game* g, int action, GameStep* out) {
    int P = g->num_players;
    int seat = g->current;
    memset(out, 0, sizeof(GameStep));
    out->seat = seat;

    if (action < 0 || action >= NUM_ACTIONS || !g->mask[action]) {
        g->invalid++;  // illegal actions are played as a pass, no penalty ...
        out->invalid = 1;
        action = ACT_PASS;
        if (g->phase != PHASE_ACT) {  // ... except in a sub-phase: force the first option
            for (int a = ACT_DISCARD; a < NUM_ACTIONS; a++) {
                if (g->mask[a]) {
                    action = a;
                    break;
                }
            }
        }
    }

    int phase = g->phase;
    int points_before = g->points[seat];
    int cards_before = g->num_bought[seat];
    apply_action(g, seat, action);  // a buy may set PHASE_NOBLE via check_nobles
    out->points = g->points[seat] - points_before;
    out->cards = g->num_bought[seat] - cards_before;

    if (phase == PHASE_NOBLE) {
        g->phase = PHASE_ACT;  // the choice was made
    }
    if (token_total(g, seat) > MAX_TOKENS) {
        g->phase = PHASE_DISCARD;  // take/reserve went over the limit
    } else if (phase == PHASE_DISCARD) {
        g->phase = PHASE_ACT;  // discarding finished
    }
    if (g->phase != PHASE_ACT) {  // same seat keeps acting; the turn is not over
        game_mask(g);
        return;
    }
    g->turn++;

    // The game only ends after the last seat of a round has acted.
    int last_seat = (g->start_seat + P - 1) % P;
    if (seat == last_seat) {
        for (int p = 0; p < P; p++) {
            if (g->points[p] >= WIN_POINTS) {
                out->by_points = 1;
                break;
            }
        }
    }
    if (!out->by_points && g->turn < g->max_turns) {
        g->current = (seat + 1) % P;
        game_mask(g);
        return;
    }

    // Most points wins, tiebreak fewest purchased cards; still tied = a draw.
    out->done = 1;
    int best_points = -1;
    int best_cards = 1000;
    for (int p = 0; p < P; p++) {
        int pts = g->points[p];
        int cards = g->num_bought[p];
        if (pts > best_points || (pts == best_points && cards < best_cards)) {
            best_points = pts;
            best_cards = cards;
        }
    }
    for (int p = 0; p < P; p++) {
        out->won[p] = (g->points[p] == best_points && g->num_bought[p] == best_cards);
        out->num_winners += out->won[p];
    }
}

static const char* COLOR_NAME[6] = {"W", "B", "G", "R", "K", "*"};
static const char* COLOR_ANSI[6] = {
    "\033[97m", "\033[94m", "\033[92m", "\033[91m", "\033[90m", "\033[93m"};

static void render_card(int card_id) {
    if (card_id < 0) {
        printf("   ....       ");
        return;
    }
    const Card* card = &CARDS[card_id];
    printf("%s%dpt %s", COLOR_ANSI[card->bonus], card->points, COLOR_NAME[card->bonus]);
    printf("\033[0m ");
    for (int c = 0; c < NUM_COLORS; c++) {
        if (card->cost[c]) {
            printf("%s%d%s%s", COLOR_ANSI[c], card->cost[c], COLOR_NAME[c], "\033[0m");
        } else {
            printf("   ");
        }
    }
    printf(" ");
}

// Plain-text board for evaluation and debugging; never called during training.
void game_render_text(Game* g) {
    int P = g->num_players;
    static const char* PHASE_NAME[3] = {"act", "discard down to 10", "choose a noble"};
    printf("\n=== Splendor  turn %d  seat %d to %s (start seat %d) ===\n",
        g->turn, g->current, PHASE_NAME[g->phase], g->start_seat);
    printf("bank:");
    for (int c = 0; c < 6; c++) {
        printf(" %s%d%s%s", COLOR_ANSI[c], g->bank[c], COLOR_NAME[c], "\033[0m");
    }
    printf("   nobles:");
    for (int i = 0; i < g->num_nobles; i++) {
        int id = g->nobles[i];
        if (id < 0) {
            printf(" [taken]");
            continue;
        }
        printf(" [");
        for (int c = 0; c < NUM_COLORS; c++) {
            if (NOBLES[id][c]) {
                printf("%s%d%s%s", COLOR_ANSI[c], NOBLES[id][c], COLOR_NAME[c], "\033[0m");
            }
        }
        printf("]");
    }
    printf("\n");
    for (int t = 2; t >= 0; t--) {
        printf("tier %d (deck %2d): ", t + 1, g->deck_n[t]);
        for (int pos = 0; pos < 4; pos++) {
            render_card(g->faceup[t][pos]);
            printf("| ");
        }
        printf("\n");
    }
    for (int p = 0; p < P; p++) {
        printf("player %d%s pts %2d cards %2d nobles %d | tokens:", p,
            (p == g->current) ? "*" : " ", g->points[p], g->num_bought[p],
            g->num_nobles_p[p]);
        for (int c = 0; c < 6; c++) {
            printf(" %s%d%s%s", COLOR_ANSI[c], g->tokens[p][c], COLOR_NAME[c], "\033[0m");
        }
        printf(" | bonus:");
        for (int c = 0; c < NUM_COLORS; c++) {
            printf(" %s%d%s%s", COLOR_ANSI[c], g->bonuses[p][c], COLOR_NAME[c], "\033[0m");
        }
        printf(" | reserved:");
        for (int r = 0; r < MAX_RESERVED; r++) {
            if (g->reserved[p][r] < 0) {
                continue;
            }
            printf(" %s", g->hidden[p][r] ? "(hidden)" : "");
            render_card(g->reserved[p][r]);
        }
        printf("\n");
    }
}

#endif  // SPLENDOR_GAME_H
