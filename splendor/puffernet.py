"""Load a PufferLib 5.0 `*_weights.bin` policy and play Splendor with it.

PufferLib 5.0 trains the C/CUDA default policy

    Linear(OBS_SIZE -> H)  ->  num_layers x MinGRU(H)  ->  Linear(H -> 73)

and exports it as a flat float32 file (`checkpoints/<env>/..._weights.bin`).
This module reproduces that network exactly (`PufferNet`), parses the file
(`load_puffernet`) and wraps the result in an `Agent` that keeps one recurrent
state per game (`PufferNetAgent`), so a 5.0 checkpoint can be evaluated with
`elo.py` / `evaluate.py` / `gui.py` alongside the 3.0 torch checkpoints.

Weight file layout -- see `make_linear` / `get_weights_aligned` / `mingru()` /
`puffernet_weight_count` in PufferLib 5.0's `src/puffercpu.c`:

    encoder weight            H x OBS_SIZE
    decoder weight            (NUM_ACTIONS + 1) x H      (last row = value head)
    mingru proj weight [0]    3H x H
    ...                       one per layer

Every block is written row-major (out_dim x in_dim, exactly torch's
`nn.Linear.weight` layout) and the running float index is rounded up to a
multiple of 8 floats after each block (the native backend aligns parameters to
16 bytes).  There are **no biases anywhere** and **no logstd** (the Splendor
action space is discrete), and the C forward pass feeds the raw observation
bytes cast to float with no scaling.
"""
import os

import numpy as np
import torch
from torch import nn

from splendor import layout as L
from splendor.agents import Agent

ALIGN = 8                      # float32s per alignment block in the weight file
DEFAULT_HIDDEN = 512
DEFAULT_LAYERS = 2
SPEC_PREFIX = 'puffer5:'       # puffer5:PATH[:HIDDEN[:LAYERS]]
MASK_FILL = -1e8               # illegal-action logit (finite, as in splendor.policy)


def _align(n):
    """Round a float index up to the next 8-float (16-byte) boundary."""
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def weight_count(obs_size, hidden, layers, num_actions=L.NUM_ACTIONS):
    """Number of float32s a `*_weights.bin` file holds for this architecture.

    Mirrors `puffernet_weight_count()` in puffercpu.c: the alignment is applied
    to the *running* index, so the count includes the padding after each block
    (the last block's padding may be missing from the file, see `load_puffernet`).
    """
    n = _align(hidden * obs_size)
    n = _align(n + (num_actions + 1) * hidden)
    for _ in range(layers):
        n = _align(n + 3 * hidden * hidden)
    return n


def read_weight_count(path):
    """Number of float32s actually stored in `path`."""
    return os.path.getsize(path) // 4


class PufferNet(nn.Module):
    """The PufferLib 5.0 default policy, one inference step at a time.

    `forward(obs, state, mask)` is a literal transcription of
    `forward_puffernet()` / `mingru()`: no observation scaling, no biases, and
    the MinGRU carry is threaded through explicitly rather than stored on the
    module, so one module can serve many independent games.
    """

    def __init__(self, obs_size, hidden, layers, num_actions=L.NUM_ACTIONS):
        super().__init__()
        self.obs_size = int(obs_size)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.num_actions = int(num_actions)
        self.encoder = nn.Linear(self.obs_size, self.hidden, bias=False)
        self.decoder = nn.Linear(self.hidden, self.num_actions + 1, bias=False)
        self.proj = nn.ModuleList([
            nn.Linear(self.hidden, 3 * self.hidden, bias=False)
            for _ in range(self.layers)])

    # --- recurrent state ---------------------------------------------------
    def zero_state(self, batch_size, device=None, dtype=None):
        """A zeroed MinGRU carry, shape (layers, batch_size, hidden)."""
        w = self.encoder.weight
        return torch.zeros(self.layers, int(batch_size), self.hidden,
                           device=device or w.device, dtype=dtype or w.dtype)

    @staticmethod
    def reset_state(state, done_mask):
        """Zero the carry of finished games, in place (C `mingru_zero_term`).

        `done_mask` is a bool/float tensor or array over the batch dimension.
        Returns `state` for convenience.
        """
        done = torch.as_tensor(np.asarray(done_mask), device=state.device)
        done = done.reshape(-1) > 0.5
        if done.any():
            state[:, done] = 0.0
        return state

    # --- forward -----------------------------------------------------------
    def forward(self, obs, state, mask=None):
        """obs: (B, obs_size) raw observation bytes as float (no scaling).

        Returns `(logits, value, state)`: logits (B, num_actions) with illegal
        actions set to -1e8, value (B,) from the last decoder output, and the
        updated carry (a new tensor; `state` is not modified).
        """
        x = self.encoder(obs)
        new_state = []
        for layer in range(self.layers):
            combined = self.proj[layer](x)
            h_in, gate, highway = combined.split(self.hidden, dim=-1)
            s = state[layer]
            gate_s = torch.sigmoid(gate)
            # h_tilde = x + 0.5 for x >= 0 else sigmoid(x) (continuous at 0)
            h_tilde = torch.where(h_in >= 0, h_in + 0.5, torch.sigmoid(h_in))
            mingru_out = s + gate_s * (h_tilde - s)
            hw_s = torch.sigmoid(highway)
            x = hw_s * mingru_out + (1.0 - hw_s) * x
            new_state.append(mingru_out)
        out = self.decoder(x)
        logits, value = out[:, :self.num_actions], out[:, self.num_actions]
        if mask is not None:
            m = torch.as_tensor(np.asarray(mask), device=logits.device) > 0
            logits = torch.where(m, logits, torch.full_like(logits, MASK_FILL))
        return logits, value, torch.stack(new_state, dim=0)


def load_puffernet(path, obs_size, hidden, layers, num_actions=L.NUM_ACTIONS):
    """Read a PufferLib 5.0 `*_weights.bin` into a `PufferNet`."""
    data = np.fromfile(path, dtype=np.float32)
    need = weight_count(obs_size, hidden, layers, num_actions)
    # The exporter may drop the padding after the final block; puffercpu.c
    # accepts the same slack (`need - file_floats <= 7 && file_floats <= need`).
    if not need - (ALIGN - 1) <= data.size <= need:
        raise ValueError(
            '%s holds %d float32s (%d bytes), but obs_size=%d hidden=%d '
            'layers=%d needs %d (%d bytes); wrong architecture or file'
            % (path, data.size, data.size * 4, obs_size, hidden, layers,
               need, need * 4))

    idx = 0

    def take(rows, cols):
        nonlocal idx
        n = rows * cols
        block = data[idx:idx + n]
        if block.size != n:
            raise ValueError('%s is truncated at float %d' % (path, idx))
        idx = _align(idx + n)
        return torch.from_numpy(block.reshape(rows, cols).copy())

    net = PufferNet(obs_size, hidden, layers, num_actions)
    with torch.no_grad():
        net.encoder.weight.copy_(take(hidden, obs_size))
        net.decoder.weight.copy_(take(num_actions + 1, hidden))
        for layer in range(layers):
            net.proj[layer].weight.copy_(take(3 * hidden, hidden))
    return net.eval()


# --- architecture guessing / spec parsing -----------------------------------
# Hidden sizes to try when only the file size is known, most likely first.
CANDIDATE_HIDDEN = (512, 256, 128, 1024, 64, 768, 384, 192, 96, 32, 16)
CANDIDATE_LAYERS = (2, 1, 3, 4, 5, 6, 8)


def infer_arch(path, num_players=None, num_actions=L.NUM_ACTIONS):
    """Guess `(num_players, hidden, layers)` from a weight file's size.

    The file is a flat blob, so the shape has to be solved for; candidates are
    tried most-likely-first and the first exact size match wins.  Returns None
    if nothing matches (i.e. it is not a PufferNet weight file).
    """
    try:
        have = read_weight_count(path)
    except OSError:
        return None
    players = (num_players,) if num_players else (2, 3, 4)
    for P in players:
        obs_size = L.obs_size(P)
        for hidden in CANDIDATE_HIDDEN:
            for layers in CANDIDATE_LAYERS:
                need = weight_count(obs_size, hidden, layers, num_actions)
                if need - (ALIGN - 1) <= have <= need:
                    return P, hidden, layers
    return None


def is_spec(spec):
    return isinstance(spec, str) and spec.strip().startswith(SPEC_PREFIX)


def parse_spec(spec):
    """'puffer5:PATH[:HIDDEN[:LAYERS]]' -> (path, hidden, layers)."""
    if not is_spec(spec):
        raise ValueError('not a PufferLib 5.0 spec: %r' % (spec,))
    parts = spec.strip()[len(SPEC_PREFIX):].split(':')
    tail = []
    while len(parts) > 1 and len(tail) < 2 and parts[-1].strip().isdigit():
        tail.insert(0, int(parts.pop()))
    hidden, layers = DEFAULT_HIDDEN, DEFAULT_LAYERS
    if len(tail) == 1:
        hidden = tail[0]
    elif len(tail) == 2:
        hidden, layers = tail
    path = ':'.join(parts).strip()
    if not path:
        raise ValueError('puffer5 needs a weight file: puffer5:PATH:HIDDEN:LAYERS')
    if hidden < 1 or layers < 1:
        raise ValueError('puffer5 hidden/layers must be positive: %r' % (spec,))
    return path, hidden, layers


def spec_for(path, hidden=DEFAULT_HIDDEN, layers=DEFAULT_LAYERS):
    return '%s%s:%d:%d' % (SPEC_PREFIX, path, hidden, layers)


def spec_label(spec):
    """'name_weights.bin · 512x2' for a puffer5 spec."""
    path, hidden, layers = parse_spec(spec)
    return '%s · %dx%d' % (os.path.basename(path), hidden, layers)


class PufferNetAgent(Agent):
    """A PufferLib 5.0 policy playing Splendor, one MinGRU state per game.

    State rows are indexed by the game index passed to `act`, grown lazily, and
    zeroed by `reset(games)` when those games finish -- the same thing the 5.0
    trainer/eval binary does with `mingru_zero_term` on the terminal flags.
    """

    def __init__(self, net, device='cpu', temperature=0.0, name=None, seed=0):
        self.net = net.to(device).eval()
        self.device = device
        self.temperature = float(temperature)
        self.name = name or 'puffer5'
        self.gen = torch.Generator(device='cpu').manual_seed(seed)
        self.state = self.net.zero_state(0, device=device)

    @classmethod
    def from_file(cls, path, num_players=2, hidden=DEFAULT_HIDDEN,
                  layers=DEFAULT_LAYERS, device='cpu', **kw):
        net = load_puffernet(path, L.obs_size(num_players), hidden, layers)
        kw.setdefault('name', os.path.basename(path))
        return cls(net, device, **kw)

    @classmethod
    def from_spec(cls, spec, num_players=2, device='cpu', **kw):
        path, hidden, layers = parse_spec(spec)
        kw.setdefault('name', spec_label(spec))
        return cls.from_file(path, num_players, hidden, layers, device, **kw)

    # --- state bookkeeping -------------------------------------------------
    def _rows(self, games, n):
        if games is None:
            return np.arange(n)
        rows = np.asarray(games, dtype=np.int64).reshape(-1)
        assert len(rows) == n, 'one game index per observation row'
        return rows

    def _grow(self, need):
        have = self.state.shape[1]
        if need <= have:
            return
        extra = self.net.zero_state(need - have, device=self.state.device)
        self.state = torch.cat([self.state, extra], dim=1)

    def act(self, obs, mask, env=None, games=None):
        obs = np.asarray(obs)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        rows = self._rows(games, obs.shape[0])
        if len(rows) == 0:
            return np.zeros(0, dtype=np.int32)
        self._grow(int(rows.max()) + 1)
        if mask is None:
            mask = L.legal_mask(obs, L.num_players_from_obs(obs.shape[1]))
        idx = torch.as_tensor(rows, device=self.state.device)
        with torch.no_grad():
            x = torch.as_tensor(obs, device=self.device).float()   # no scaling
            logits, _, new_state = self.net(x, self.state[:, idx], mask)
            self.state[:, idx] = new_state
            if self.temperature > 0:
                probs = torch.softmax(logits.float() / self.temperature, dim=1)
                a = torch.multinomial(probs.cpu(), 1, generator=self.gen)[:, 0]
            else:
                a = logits.argmax(dim=1).cpu()
        return a.numpy().astype(np.int32)

    def reset(self, games=None):
        if games is None:
            self.state.zero_()
            return
        rows = np.asarray(games, dtype=np.int64).reshape(-1)
        if len(rows) == 0:
            return
        self._grow(int(rows.max()) + 1)
        self.state[:, torch.as_tensor(rows, device=self.state.device)] = 0.0
