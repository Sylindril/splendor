"""Batched PUCT / AlphaZero search over Splendor states.

The environment is the model: a scratch `Splendor(num_envs=batch)` provides the
transitions.  Root `b` owns scratch game `b`; a node is expanded by restoring
its parent's 400-byte snapshot into that slot, stepping the chosen action and
reading back the new snapshot, observations, mask and rewards.

Multiplayer is handled by storing a *per-seat* value vector in every node: the
network is evaluated on all `P` observation rows of a leaf, so backups do not
have to assume a zero-sum two-player game.  Selection uses `Q` from the point of
view of the seat the environment is waiting on at that node (which may be the
same seat twice in a row during a discard / noble sub-phase - those are ordinary
decision nodes here).

Terminal leaves are never expanded.  A step that ends the game immediately deals
a new one inside the C env, so the observations it returns belong to a different
game; the only usable signal is the rewards vector, which is clipped to [-1, 1]
and used as the node's value for every seat.

Hidden information: `determinize=True` re-deals what the seat to move cannot see
(deck order and opponents' deck-top reserves) *once*, before the root snapshot is
taken.  The env RNG is part of that snapshot, so the whole search explores a
single determinization rather than re-sampling one per simulation.
"""
import numpy as np

from splendor import layout as L
from splendor.agents import Agent, load_policy
from splendor.splendor import Splendor

A = L.NUM_ACTIONS


def _masked_softmax(logits, mask):
    """Row-wise softmax restricted to `mask` (bool); illegal entries are 0."""
    logits = np.where(mask, logits, -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p[~mask] = 0.0
    total = p.sum(axis=1, keepdims=True)
    # A row with no legal action cannot happen (PASS is always legal), but be safe.
    return np.where(total > 0, p / np.maximum(total, 1e-12), mask / np.maximum(mask.sum(1, keepdims=True), 1))


class MCTS:
    """PUCT search over a batch of root states.

    `search(states, seats=None) -> (visits (B, 72) int32, values (B, P) float32)`
    where `states` are snapshots from `env.get_state(i)`.
    """

    def __init__(self, policy, num_players=2, sims=200, c_puct=1.5,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25, determinize=True,
                 noise=False, device='cpu', seed=0, capacity=64, max_turns=None):
        import torch
        self.torch = torch
        if isinstance(policy, str):
            policy = load_policy(policy, num_players, device)
        self.policy = policy.to(device).eval()
        self.device = device
        self.P = num_players
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.determinize = determinize
        self.noise = noise
        self.max_turns = max_turns
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.env = None
        self.cap = 0
        self._alloc(max(1, capacity))

    # -- storage --------------------------------------------------------------
    def _alloc(self, cap):
        """(Re)allocate the scratch env and the node arrays for `cap` roots."""
        if self.env is not None:
            self.env.close()
        self.cap = cap
        self.env = Splendor(num_envs=cap, num_players=self.P, seed=self.seed,
                            max_turns=self.max_turns, report_interval=10**9)
        self.env.reset(seed=self.seed)
        self.act_buf = np.full(self.env.num_agents, L.NOOP, dtype=np.int32)
        M = self.sims + 2               # root + one new node per simulation
        P = self.P
        self.M = M
        self.N = np.zeros((cap, M), dtype=np.int32)
        self.W = np.zeros((cap, M, P), dtype=np.float32)
        self.prior = np.zeros((cap, M, A), dtype=np.float32)
        self.child = np.full((cap, M, A), -1, dtype=np.int32)
        self.mask = np.zeros((cap, M, A), dtype=bool)
        self.tomove = np.zeros((cap, M), dtype=np.int64)
        self.term = np.zeros((cap, M), dtype=bool)
        self.tval = np.zeros((cap, M, P), dtype=np.float32)
        self.states = [[None] * M for _ in range(cap)]
        self.n_nodes = np.zeros(cap, dtype=np.int64)

    def _ensure(self, batch, sims):
        if batch > self.cap or sims + 2 > self.M:
            self.sims = max(sims, self.sims)
            self._alloc(max(batch, self.cap))

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    # -- network --------------------------------------------------------------
    def _evaluate(self, obs):
        """obs (n, OBS_N) uint8 -> (logits (n, A) float32, value (n,) in [-1, 1])."""
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(obs)).to(self.device)
            logits, value = self.policy.forward_eval(x, {})
            logits = logits.float().cpu().numpy()
            value = value.float().cpu().numpy().reshape(-1)
        return logits, np.clip(value, -1.0, 1.0)

    # -- search ---------------------------------------------------------------
    def search(self, states, seats=None, sims=None, noise=None):
        sims = self.sims if sims is None else sims
        noise = self.noise if noise is None else noise
        B = len(states)
        self._ensure(B, sims)
        env, P = self.env, self.P
        tm_off = L.to_move_offset(P)
        mask_lo, mask_hi = L.mask_offset(P), L.turn_offset(P)

        # ---- reset the node arrays for the rows we use ----
        b_idx = np.arange(B)
        self.N[:B] = 0
        self.W[:B] = 0.0
        self.child[:B] = -1
        self.mask[:B] = False
        self.term[:B] = False
        self.n_nodes[:B] = 1

        # ---- root: restore, optionally determinize once, then evaluate ----
        for b in range(B):
            env.put_state(b, state=states[b])
        obs = env.observations
        if seats is None:
            seats = obs[:, tm_off].reshape(-1, P)[:B].argmax(axis=1)
        seats = np.asarray(seats, dtype=np.int64)
        if self.determinize:
            for b in range(B):
                env.put_state(b, determinize=int(seats[b]))
        rows = (b_idx[:, None] * P + np.arange(P)[None, :]).ravel()
        root_obs = obs[rows].copy()
        for b in range(B):
            self.states[b][0] = env.get_state(b)

        logits, value = self._evaluate(root_obs)
        self.W[:B, 0] = value.reshape(B, P)
        self.N[:B, 0] = 1
        self.tomove[:B, 0] = seats
        rmask = root_obs.reshape(B, P, -1)[b_idx, seats, mask_lo:mask_hi] > 0
        self.mask[:B, 0] = rmask
        pri = _masked_softmax(logits.reshape(B, P, A)[b_idx, seats], rmask)
        if noise:
            for b in range(B):
                legal = np.flatnonzero(rmask[b])
                if len(legal) > 1:
                    d = self.rng.dirichlet([self.dirichlet_alpha] * len(legal))
                    pri[b, legal] = (1 - self.dirichlet_eps) * pri[b, legal] \
                        + self.dirichlet_eps * d
        self.prior[:B, 0] = pri

        # ---- simulations ----
        for _ in range(sims):
            self._simulate(B)

        visits = np.zeros((B, A), dtype=np.int32)
        ch = self.child[:B, 0]
        ok = ch >= 0
        visits[ok] = self.N[:B][np.nonzero(ok)[0], ch[ok]]
        values = self.W[:B, 0] / np.maximum(self.N[:B, 0], 1)[:, None]
        return visits, values.astype(np.float32)

    def _simulate(self, B):
        env, P = self.env, self.P
        N, W, prior, child, mask = self.N, self.W, self.prior, self.child, self.mask
        cur = np.zeros(B, dtype=np.int64)
        alive = np.arange(B)
        path = []                      # [(root indices, node indices)] per depth
        exp_b, exp_par, exp_act = [], [], []
        hit_b, hit_node = [], []
        for _depth in range(4 * self.M + 16):
            if alive.size == 0:
                break
            n = cur[alive]
            is_t = self.term[alive, n]
            if is_t.any():
                t = np.flatnonzero(is_t)
                hit_b.append(alive[t])
                hit_node.append(n[t])
                keep = ~is_t
                alive, n = alive[keep], n[keep]
                if alive.size == 0:
                    break
            path.append((alive, n))
            s = self.tomove[alive, n]
            Nn = N[alive, n]
            ch = child[alive, n]                       # (k, A)
            chc = np.maximum(ch, 0)
            rows = alive[:, None]
            cN = np.where(ch >= 0, N[rows, chc], 0)
            cW = np.take_along_axis(W[rows, chc], s[:, None, None], axis=2)[:, :, 0]
            pq = W[alive, n, s] / np.maximum(Nn, 1)
            Q = np.where(cN > 0, cW / np.maximum(cN, 1), pq[:, None])
            U = (self.c_puct * prior[alive, n]
                 * np.sqrt(np.maximum(Nn, 1)).astype(np.float32)[:, None] / (1 + cN))
            score = np.where(mask[alive, n], Q + U, -np.inf)
            a = score.argmax(axis=1)
            c = ch[np.arange(a.size), a]
            new = c < 0
            if new.any():
                k = np.flatnonzero(new)
                exp_b.append(alive[k])
                exp_par.append(n[k])
                exp_act.append(a[k])
            go = np.flatnonzero(~new)
            cur[alive[go]] = c[go]
            alive = alive[go]

        v = np.zeros((B, P), dtype=np.float32)
        done = np.zeros(B, dtype=bool)
        if hit_b:
            hb = np.concatenate(hit_b)
            hn = np.concatenate(hit_node)
            v[hb] = self.tval[hb, hn]
            done[hb] = True
            N[hb, hn] += 1
            W[hb, hn] += v[hb]

        if exp_b:
            eb = np.concatenate(exp_b)
            ep = np.concatenate(exp_par)
            ea = np.concatenate(exp_act)
            self.act_buf[:] = L.NOOP
            for i in range(eb.size):
                b = int(eb[i])
                env.put_state(b, state=self.states[b][ep[i]])
                self.act_buf[b * P + int(self.tomove[b, ep[i]])] = ea[i]
            env.step(self.act_buf)

            obs = env.observations
            terms = env.terminals.reshape(-1, P)[:, 0].astype(bool)
            rew = env.rewards.reshape(-1, P)
            new_nodes = self.n_nodes[eb]
            self.n_nodes[eb] = new_nodes + 1
            child[eb, ep, ea] = new_nodes.astype(np.int32)

            fin = terms[eb]
            if fin.any():
                fb, fn = eb[fin], new_nodes[fin]
                tv = np.clip(rew[fb], -1.0, 1.0).astype(np.float32)
                self.term[fb, fn] = True
                self.tval[fb, fn] = tv
                N[fb, fn] = 1
                W[fb, fn] = tv
                v[fb] = tv
                done[fb] = True

            live = ~fin
            if live.any():
                lb, ln = eb[live], new_nodes[live]
                rows = (lb[:, None] * P + np.arange(P)[None, :]).ravel()
                leaf_obs = obs[rows].copy()
                for b in lb:
                    self.states[int(b)][self.n_nodes[b] - 1] = env.get_state(int(b))
                logits, value = self._evaluate(leaf_obs)
                k = lb.size
                lo, hi = L.mask_offset(P), L.turn_offset(P)
                o3 = leaf_obs.reshape(k, P, -1)
                ls = o3[:, :, L.to_move_offset(P)].argmax(axis=1)
                m = o3[np.arange(k), ls, lo:hi] > 0
                self.tomove[lb, ln] = ls
                self.mask[lb, ln] = m
                self.term[lb, ln] = False
                self.prior[lb, ln] = _masked_softmax(logits.reshape(k, P, A)[np.arange(k), ls], m)
                vv = value.reshape(k, P).astype(np.float32)
                N[lb, ln] = 1
                W[lb, ln] = vv
                v[lb] = vv
                done[lb] = True

        for idx, node in path:
            N[idx, node] += 1
            W[idx, node] += v[idx]


class MCTSAgent(Agent):
    """`splendor.agents.Agent` that plays the argmax (or a temperature sample)
    of the root visit counts of a PUCT search."""

    name = 'mcts'

    def __init__(self, policy, num_players=2, sims=100, temperature=0.0,
                 determinize=True, noise=False, c_puct=1.5, device='cpu',
                 seed=0, name=None, capacity=64, max_turns=None):
        self.P = num_players
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.mcts = MCTS(policy, num_players, sims=sims, c_puct=c_puct,
                         determinize=determinize, noise=noise, device=device,
                         seed=seed, capacity=capacity, max_turns=max_turns)
        self.name = name or f'mcts{sims}'

    @property
    def sims(self):
        return self.mcts.sims

    def reset(self, games=None):
        pass

    def act(self, obs, mask, env=None, games=None):
        if env is None or games is None:
            raise ValueError('MCTSAgent needs the env and the game indices')
        games = np.asarray(games)
        P = self.P
        tm = env.observations[:, L.to_move_offset(P)].reshape(-1, P)
        seats = tm[games].argmax(axis=1)
        states = [env.get_state(int(g)) for g in games]
        visits, _ = self.mcts.search(states, seats)
        return self.pick(visits, mask)

    def pick(self, visits, mask=None):
        out = np.empty(len(visits), dtype=np.int32)
        for i, v in enumerate(visits):
            if v.sum() == 0:                       # sims == 0 or nothing legal
                legal = np.flatnonzero(mask[i]) if mask is not None else [L.PASS]
                out[i] = legal[0] if len(legal) else L.PASS
            elif self.temperature > 0:
                p = v.astype(np.float64) ** (1.0 / self.temperature)
                out[i] = self.rng.choice(len(v), p=p / p.sum())
            else:
                out[i] = int(np.argmax(v))
        return out

    def close(self):
        self.mcts.close()


def benchmark(sims=100, batch=64, num_players=2, device='cpu', rounds=3,
              checkpoint='latest'):
    """Root-simulations per second for a batch of `batch` independent roots."""
    import time
    net = load_policy(checkpoint, num_players, device)
    env = Splendor(num_envs=batch, num_players=num_players, report_interval=10**9)
    env.reset(seed=0)
    states = [env.get_state(i) for i in range(batch)]
    m = MCTS(net, num_players, sims=sims, device=device, capacity=batch)
    m.search(states[:4], sims=4)          # warm up
    best = 0.0
    for _ in range(rounds):
        t = time.perf_counter()
        m.search(states)
        dt = time.perf_counter() - t
        best = max(best, batch * sims / dt)
    print(f'{device:>4}  batch={batch} sims={sims} P={num_players}: '
          f'{best:,.0f} root-simulations/s ({best/batch:,.0f} per root)')
    m.close()
    env.close()
    return best


if __name__ == '__main__':
    import sys
    devs = sys.argv[1:] or ['cpu', 'mps']
    for d in devs:
        benchmark(device=d)
