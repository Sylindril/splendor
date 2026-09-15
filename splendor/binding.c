#include <stddef.h>
#include "splendor.h"

#define Env Splendor
#define MY_GET
#define MY_PUT
#include "env_binding.h"

// Opaque snapshot of the whole game state (everything after the buffer pointers).
#define STATE_OFFSET offsetof(Splendor, num_players)
#define STATE_SIZE (sizeof(Splendor) - STATE_OFFSET)

static PyObject* my_get(PyObject* dict, Env* env) {
    PyObject* state = PyBytes_FromStringAndSize((char*)env + STATE_OFFSET, STATE_SIZE);
    PyDict_SetItemString(dict, "state", state);
    Py_DECREF(state);
    return dict;
}

// env_put(handle, state=bytes) restores a snapshot from env_get;
// env_put(handle, bonuses=[P*5 ints]) overwrites bonus cards (tests / debugging);
// env_put(handle, determinize=seat) re-deals what `seat` cannot see.
// Either way the mask and observations are recomputed.
static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    PyObject* state = PyDict_GetItemString(kwargs, "state");
    if (state != NULL) {
        if (!PyBytes_Check(state) || PyBytes_Size(state) != (Py_ssize_t)STATE_SIZE) {
            PyErr_SetString(PyExc_ValueError, "state must be bytes from env_get");
            return 1;
        }
        memcpy((char*)env + STATE_OFFSET, PyBytes_AsString(state), STATE_SIZE);
    }
    PyObject* bonuses = PyDict_GetItemString(kwargs, "bonuses");
    if (bonuses != NULL) {
        int n = env->num_players * NUM_COLORS;
        if (!PySequence_Check(bonuses) || PySequence_Size(bonuses) != n) {
            PyErr_SetString(PyExc_ValueError, "bonuses must be a flat list of num_players*5 ints");
            return 1;
        }
        for (int i = 0; i < n; i++) {
            PyObject* item = PySequence_GetItem(bonuses, i);
            env->bonuses[i / NUM_COLORS][i % NUM_COLORS] = (unsigned char)PyLong_AsLong(item);
            Py_DECREF(item);
        }
    }
    PyObject* det = PyDict_GetItemString(kwargs, "determinize");
    if (det != NULL) {
        int seat = (int)PyLong_AsLong(det);
        if (seat < 0 || seat >= env->num_players) {
            PyErr_SetString(PyExc_ValueError, "determinize must be a seat index");
            return 1;
        }
        determinize(env, seat);
    }
    compute_mask(env);
    write_obs(env);
    return 0;
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;
    int num_players = (int)unpack(kwargs, "num_players");
    if (PyErr_Occurred()) {
        return 1;
    }
    if (num_players < 2 || num_players > MAX_PLAYERS) {
        PyErr_SetString(PyExc_ValueError, "num_players must be between 2 and 4");
        return 1;
    }
    env->num_players = num_players;
    env->max_turns = (int)unpack(kwargs, "max_turns");
    env->reward_point = (float)unpack(kwargs, "reward_point");
    env->reward_card = (float)unpack(kwargs, "reward_card");
    env->reward_win = (float)unpack(kwargs, "reward_win");
    env->reward_loss = (float)unpack(kwargs, "reward_loss");
    env->auto_reset = (int)unpack(kwargs, "auto_reset");
    uint64_t seed = (uint64_t)(int64_t)unpack(kwargs, "seed");
    if (PyErr_Occurred()) {
        return 1;
    }
    if (env->max_turns <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_turns must be positive");
        return 1;
    }
    env->rng = seed_rng(seed);
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "points", log->points);
    assign_to_dict(dict, "cards", log->cards);
    assign_to_dict(dict, "nobles", log->nobles);
    assign_to_dict(dict, "game_length", log->game_length);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "invalid", log->invalid);
    return 0;
}
