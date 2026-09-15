// Parity fixture generator for splendor/puffernet.py.
//
// Includes PufferLib 5.0's src/puffercpu.c (without PUFFERCPU_EVAL_MAIN, so no
// env header / raylib is needed), writes a random weights file of exactly the
// size puffernet_weight_count() expects for (OBS, HIDDEN, LAYERS), then runs
// forward_puffernet() for STEPS consecutive steps on random observations,
// masks and terminals, dumping the decoder outputs (72 logits + value).
//
// build: clang -O1 -I$PL5/src gen_fixture.c -o gen_fixture -lm
#include "puffercpu.c"

#define OBS    336     // splendor, 2 players: 240 + 48*2
#define HIDDEN 64
#define LAYERS 2
#define BATCH  4
#define STEPS  6
#define NATN   72

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static uint64_t next_u64(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static int next_int(int n) { return (int)(next_u64() % (uint64_t)n); }

static int align8(int n) { return (n + 7) & ~7; }

// Exactly puffernet_weight_count() from puffercpu.c (it lives inside
// #ifdef PUFFERCPU_EVAL_MAIN, so it is repeated here).
static int weight_count(int input_dim, int hidden, int layers, int atn_sum) {
    int n = 0;
    n = align8(n + hidden*input_dim);
    n = align8(n + (atn_sum + 1)*hidden);
    for (int l = 0; l < layers; l++) {
        n = align8(n + 3*hidden*hidden);
    }
    return n;
}

static void dump(const char* name, const void* data, size_t bytes) {
    FILE* f = fopen(name, "wb");
    if (!f) { perror(name); exit(1); }
    fwrite(data, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const char* dir = (argc > 1) ? argv[1] : ".";
    char path[2048];

    int atn_sum = NATN;
    int n_weights = weight_count(OBS, HIDDEN, LAYERS, atn_sum);
    float* w = calloc(n_weights, sizeof(float));
    // Small weights keep the float32 sums well conditioned so that the torch
    // matmul (different summation order / FMA) matches to ~1e-5.
    for (int i = 0; i < n_weights; i++) {
        w[i] = (float)(next_int(2001) - 1000) * 1e-4f;   // in [-0.1, 0.1]
    }
    snprintf(path, sizeof(path), "%s/puffernet_336x64x2_weights.bin", dir);
    dump(path, w, (size_t)n_weights*sizeof(float));

    Weights* weights = load_weights(path);
    assert(weights != NULL);
    int act_sizes[1] = {NATN};
    PufferNet* net = make_puffernet(weights, BATCH, OBS, HIDDEN, LAYERS, act_sizes, 1);
    assert(net->is_continuous == 0);

    unsigned char* obs = calloc((size_t)STEPS*BATCH*OBS, 1);
    unsigned char* masks = calloc((size_t)STEPS*BATCH*NATN, 1);
    float* terms = calloc((size_t)STEPS*BATCH, sizeof(float));
    float* out = calloc((size_t)STEPS*BATCH*(NATN + 1), sizeof(float));

    for (int s = 0; s < STEPS; s++) {
        for (int i = 0; i < BATCH*OBS; i++) {
            // Splendor observation bytes are small (counts, one-hots, a mask);
            // only the turn byte gets large. Keep the same rough scale.
            obs[s*BATCH*OBS + i] = (unsigned char)next_int(16);
        }
        for (int b = 0; b < BATCH; b++) {
            unsigned char* m = masks + (s*BATCH + b)*NATN;
            int legal = 0;
            for (int a = 0; a < NATN; a++) {
                m[a] = (next_int(4) == 0);
                legal += m[a];
            }
            if (!legal) m[60] = 1;                  // PASS, as the env guarantees
        }
        // Terminals seen by the forward pass (written by the previous step):
        // step 3 ends games 1 and 2, step 5 ends game 0.
        if (s == 3) { terms[s*BATCH + 1] = 1.0f; terms[s*BATCH + 2] = 1.0f; }
        if (s == 5) { terms[s*BATCH + 0] = 1.0f; }
    }

    float* obs_f = calloc(BATCH*OBS, sizeof(float));
    float actions[BATCH];
    for (int s = 0; s < STEPS; s++) {
        for (int i = 0; i < BATCH*OBS; i++) {
            obs_f[i] = obs[s*BATCH*OBS + i];
        }
        forward_puffernet(net, obs_f, actions, masks + (size_t)s*BATCH*NATN,
            terms + (size_t)s*BATCH);
        memcpy(out + (size_t)s*BATCH*(NATN + 1), net->decoder->output,
            (size_t)BATCH*(NATN + 1)*sizeof(float));
    }

    snprintf(path, sizeof(path), "%s/puffernet_336x64x2_obs.bin", dir);
    dump(path, obs, (size_t)STEPS*BATCH*OBS);
    snprintf(path, sizeof(path), "%s/puffernet_336x64x2_mask.bin", dir);
    dump(path, masks, (size_t)STEPS*BATCH*NATN);
    snprintf(path, sizeof(path), "%s/puffernet_336x64x2_term.bin", dir);
    dump(path, terms, (size_t)STEPS*BATCH*sizeof(float));
    snprintf(path, sizeof(path), "%s/puffernet_336x64x2_out.bin", dir);
    dump(path, out, (size_t)STEPS*BATCH*(NATN + 1)*sizeof(float));

    printf("weights floats=%d bytes=%zu\n", n_weights, (size_t)n_weights*sizeof(float));
    printf("steps=%d batch=%d obs=%d hidden=%d layers=%d out_dim=%d\n",
        STEPS, BATCH, OBS, HIDDEN, LAYERS, NATN + 1);
    for (int s = 0; s < STEPS; s++) {
        for (int b = 0; b < BATCH; b++) {
            float* o = out + ((size_t)s*BATCH + b)*(NATN + 1);
            printf("step %d agent %d logits[0..3]=%.9g %.9g %.9g %.9g value=%.9g\n",
                s, b, o[0], o[1], o[2], o[3], o[NATN]);
        }
    }
    return 0;
}
