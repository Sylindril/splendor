"""Tests for splendor/puffernet.py (PufferLib 5.0 `*_weights.bin` loader).

The parity fixture in tests/fixtures/ was produced by
`tests/fixtures/puffernet_fixture.c`, which #includes PufferLib 5.0's
`src/puffercpu.c` and runs the real `forward_puffernet()`:

    clang -O1 -I$PL5/src tests/fixtures/puffernet_fixture.c -o gen -lm
    ./gen tests/fixtures

It writes a random weight file of exactly `puffernet_weight_count(336, 64, 2)`
floats plus the observations / masks / terminals it fed the net and the raw
decoder outputs (72 logits + value per agent) of six consecutive steps.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from splendor import agents as A
from splendor import layout as L
from splendor import puffernet as PN

FIXTURES = os.path.join(ROOT, 'tests', 'fixtures')
OBS_SIZE, HIDDEN, LAYERS = 336, 64, 2     # 2-player Splendor: 240 + 48*2
STEPS, BATCH = 6, 4
WEIGHTS = os.path.join(FIXTURES, 'puffernet_336x64x2_weights.bin')


def fixture(name, dtype):
    return np.fromfile(os.path.join(FIXTURES, 'puffernet_336x64x2_%s.bin' % name),
                       dtype=dtype)


needs_fixture = pytest.mark.skipif(
    not os.path.isfile(WEIGHTS), reason='parity fixture not built')


# --- weight file layout -----------------------------------------------------
def test_weight_count_matches_the_c_formula():
    """align8 after each block, no biases, no logstd (72 discrete actions)."""
    def align8(n):
        return (n + 7) & ~7
    want = align8(HIDDEN * OBS_SIZE)
    want = align8(want + (L.NUM_ACTIONS + 1) * HIDDEN)
    for _ in range(LAYERS):
        want = align8(want + 3 * HIDDEN * HIDDEN)
    assert PN.weight_count(OBS_SIZE, HIDDEN, LAYERS) == want
    # 512x2 for the 2-player env, the shipped default
    assert PN.weight_count(L.obs_size(2), 512, 2) == (
        align8(512 * 336) + align8(73 * 512) + 2 * align8(3 * 512 * 512))


@needs_fixture
def test_fixture_file_has_exactly_the_expected_size():
    assert PN.read_weight_count(WEIGHTS) == PN.weight_count(
        OBS_SIZE, HIDDEN, LAYERS)


@needs_fixture
def test_wrong_architecture_is_a_clear_error():
    for obs_size, hidden, layers in ((OBS_SIZE, HIDDEN, 3),
                                     (OBS_SIZE, 128, LAYERS),
                                     (L.obs_size(4), HIDDEN, LAYERS)):
        with pytest.raises(ValueError, match='needs'):
            PN.load_puffernet(WEIGHTS, obs_size, hidden, layers)


def test_truncated_file_is_a_clear_error(tmp_path):
    path = str(tmp_path / 'short_weights.bin')
    np.zeros(PN.weight_count(OBS_SIZE, HIDDEN, LAYERS) - 64,
             dtype=np.float32).tofile(path)
    with pytest.raises(ValueError, match='needs'):
        PN.load_puffernet(path, OBS_SIZE, HIDDEN, LAYERS)


def test_no_biases_anywhere():
    net = PN.PufferNet(OBS_SIZE, HIDDEN, LAYERS)
    assert net.encoder.weight.shape == (HIDDEN, OBS_SIZE)
    assert net.decoder.weight.shape == (L.NUM_ACTIONS + 1, HIDDEN)
    assert all(p.weight.shape == (3 * HIDDEN, HIDDEN) for p in net.proj)
    assert all(b is None for b in [net.encoder.bias, net.decoder.bias]
               + [p.bias for p in net.proj])
    assert len(list(net.parameters())) == 2 + LAYERS


def test_missing_trailing_padding_is_accepted(tmp_path):
    """puffercpu.c allows the file to stop at the last weight, before the
    8-float padding that `puffernet_weight_count` adds after every block."""
    obs_size, hidden, layers = 10, 6, 1      # 3*6*6 = 108 -> 4 floats of padding
    need = PN.weight_count(obs_size, hidden, layers)
    path = str(tmp_path / 'unpadded_weights.bin')
    np.arange(need - 4, dtype=np.float32).tofile(path)   # 4 floats of padding
    net = PN.load_puffernet(path, obs_size, hidden, layers)
    assert net.proj[0].weight.shape == (3 * hidden, hidden)
    assert net.proj[0].weight[-1, -1].item() == need - 5


@needs_fixture
def test_blocks_are_read_at_the_aligned_offsets():
    """Each block starts at the 8-float boundary after the previous one."""
    data = np.fromfile(WEIGHTS, dtype=np.float32)
    net = PN.load_puffernet(WEIGHTS, OBS_SIZE, HIDDEN, LAYERS)
    off = 0
    for want in [net.encoder.weight, net.decoder.weight] + \
                [p.weight for p in net.proj]:
        n = want.numel()
        got = data[off:off + n].reshape(tuple(want.shape))
        assert np.array_equal(got, want.detach().numpy())
        off = (off + n + 7) & ~7


# --- parity with the C forward pass -----------------------------------------
@needs_fixture
def test_matches_c_forward_over_six_steps_with_state_carry():
    """Same weights, obs, masks and terminals -> same decoder output.

    Steps 3 and 5 carry terminal flags, so this also covers `reset_state`
    (the C `mingru_zero_term`) zeroing only the finished rows' carry.
    """
    obs = fixture('obs', np.uint8).reshape(STEPS, BATCH, OBS_SIZE)
    mask = fixture('mask', np.uint8).reshape(STEPS, BATCH, L.NUM_ACTIONS)
    term = fixture('term', np.float32).reshape(STEPS, BATCH)
    want = fixture('out', np.float32).reshape(STEPS, BATCH, L.NUM_ACTIONS + 1)
    assert term[3].sum() == 2 and term[5].sum() == 1, 'fixture lost its resets'

    net = PN.load_puffernet(WEIGHTS, OBS_SIZE, HIDDEN, LAYERS)
    state = net.zero_state(BATCH)
    for step in range(STEPS):
        net.reset_state(state, term[step])
        x = torch.from_numpy(obs[step]).float()
        with torch.no_grad():
            # the C decoder output is unmasked; masking happens in multidiscrete
            raw, value, next_state = net(x, state)
            logits = net(x, state, mask[step])[0]
        state = next_state
        np.testing.assert_allclose(raw.numpy(), want[step, :, :L.NUM_ACTIONS],
                                   atol=1e-4, rtol=1e-4,
                                   err_msg='logits differ at step %d' % step)
        np.testing.assert_allclose(value.numpy(), want[step, :, L.NUM_ACTIONS],
                                   atol=1e-4, rtol=1e-4,
                                   err_msg='value differs at step %d' % step)
        legal = mask[step] > 0
        assert (logits.numpy()[~legal] == PN.MASK_FILL).all()
        assert np.array_equal(logits.numpy()[legal], raw.numpy()[legal])


@needs_fixture
def test_state_carry_actually_matters():
    """Without the carried state the outputs would not match (sanity check)."""
    obs = fixture('obs', np.uint8).reshape(STEPS, BATCH, OBS_SIZE)
    want = fixture('out', np.float32).reshape(STEPS, BATCH, L.NUM_ACTIONS + 1)
    net = PN.load_puffernet(WEIGHTS, OBS_SIZE, HIDDEN, LAYERS)
    with torch.no_grad():   # step 1 from a zero state instead of step 0's carry
        _, value, _ = net(torch.from_numpy(obs[1]).float(), net.zero_state(BATCH))
    assert not np.allclose(value.numpy(), want[1, :, L.NUM_ACTIONS], atol=1e-3)


@needs_fixture
def test_reset_state_only_clears_the_finished_rows():
    net = PN.load_puffernet(WEIGHTS, OBS_SIZE, HIDDEN, LAYERS)
    state = torch.randn(LAYERS, BATCH, HIDDEN)
    before = state.clone()
    net.reset_state(state, np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32))
    assert (state[:, 1] == 0).all() and (state[:, 3] == 0).all()
    assert torch.equal(state[:, 0], before[:, 0])
    assert torch.equal(state[:, 2], before[:, 2])


# --- spec parsing -----------------------------------------------------------
@pytest.mark.parametrize('spec, want', [
    ('puffer5:a_weights.bin', ('a_weights.bin', 512, 2)),
    ('puffer5:a_weights.bin:256', ('a_weights.bin', 256, 2)),
    ('puffer5:models/a_weights.bin:128:3', ('models/a_weights.bin', 128, 3)),
])
def test_parse_spec(spec, want):
    assert PN.parse_spec(spec) == want


def test_spec_label_and_round_trip():
    spec = PN.spec_for('models/splendor_weights.bin', 512, 2)
    assert spec == 'puffer5:models/splendor_weights.bin:512:2'
    assert PN.spec_label(spec) == 'splendor_weights.bin · 512x2'
    assert PN.parse_spec(spec)[0] == 'models/splendor_weights.bin'
    assert not PN.is_spec('random') and not PN.is_spec('models/x.pt')
    assert PN.is_spec(spec) and PN.is_spec('  puffer5:x  ')
    for bad in ('random', 'models/x.pt', 'puffer5:', 'puffer5:  ', 'puffer5:x:0'):
        with pytest.raises(ValueError):
            PN.parse_spec(bad)


@needs_fixture
def test_infer_arch_from_file_size():
    assert PN.infer_arch(WEIGHTS) == (2, HIDDEN, LAYERS)
    assert PN.infer_arch(WEIGHTS, num_players=4) is None
    assert PN.infer_arch(os.path.join(FIXTURES, 'puffernet_336x64x2_obs.bin')) is None


# --- the agent --------------------------------------------------------------
@needs_fixture
def test_agent_from_spec_plays_legal_moves_and_resets_state():
    """20 games vs random-legal: every action legal, state reset on game end."""
    spec = PN.spec_for(os.path.relpath(WEIGHTS, ROOT), HIDDEN, LAYERS)
    agent = A.make_agent(spec, num_players=2, seed=0)
    assert isinstance(agent, PN.PufferNetAgent)
    assert agent.name == PN.spec_label(spec)

    seen = {'rows': 0}
    illegal = []

    class Checked(A.Agent):
        """Wraps the agent so play_games' calls can be inspected."""
        name = 'checked'

        def act(self, obs, mask, env=None, games=None):
            actions = agent.act(obs, mask, env, games)
            assert actions.dtype == np.int32
            if not mask[np.arange(len(actions)), actions].all():
                illegal.append(actions)
            seen['rows'] += len(actions)
            # every game we have acted for must have a state row
            assert agent.state.shape[1] > int(np.max(games))
            return actions

        def reset(self, games=None):
            agent.reset(games)

    result = A.play_games([Checked(), A.RandomAgent(1)], 20, 2, seed=3)
    assert result['finished'].all()
    assert not illegal, 'puffer5 agent played an illegal action'
    assert seen['rows'] > 100
    # play_games resets every agent when a game finishes; all 20 games did
    assert agent.state.shape[1] >= 20
    assert float(agent.state.abs().sum()) == 0.0


@needs_fixture
def test_agent_state_is_per_game_and_grows_lazily():
    spec = PN.spec_for(WEIGHTS, HIDDEN, LAYERS)
    agent = A.make_agent(spec, num_players=2, seed=0)
    assert agent.state.shape == (LAYERS, 0, HIDDEN)
    obs = np.zeros((2, OBS_SIZE), dtype=np.uint8)
    mask = np.zeros((2, L.NUM_ACTIONS), dtype=np.uint8)
    mask[:, L.PASS] = 1
    a = agent.act(obs, mask, None, np.array([0, 7]))
    assert (a == L.PASS).all()
    assert agent.state.shape == (LAYERS, 8, HIDDEN)
    assert float(agent.state[:, 1:7].abs().sum()) == 0.0   # untouched games
    assert float(agent.state[:, 7].abs().sum()) > 0.0
    agent.reset([7])
    assert float(agent.state[:, 7].abs().sum()) == 0.0
    assert float(agent.state[:, 0].abs().sum()) > 0.0


@needs_fixture
def test_agent_samples_when_temperature_is_set():
    net = PN.load_puffernet(WEIGHTS, OBS_SIZE, HIDDEN, LAYERS)
    agent = PN.PufferNetAgent(net, temperature=1.0, seed=0)
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 16, (64, OBS_SIZE), dtype=np.uint8)
    mask = np.zeros((64, L.NUM_ACTIONS), dtype=np.uint8)
    mask[:, [3, 11, 42]] = 1
    a = agent.act(obs, mask, None, np.arange(64))
    assert set(np.unique(a)) <= {3, 11, 42}
    assert len(np.unique(a)) > 1, 'temperature 1.0 should not be deterministic'


# --- gui.py wiring ----------------------------------------------------------
@needs_fixture
def test_gui_lists_weight_files_and_builds_the_agent():
    import gui

    found = gui.puffer5_checkpoints(roots=(FIXTURES,))
    entry = [c for c in found if 'puffernet_336x64x2' in c['path']]
    assert len(entry) == 1, found
    entry = entry[0]
    assert entry['kind'] == 'puffer5'
    assert entry['num_players'] == 2
    assert entry['label'] == 'puffernet_336x64x2_weights.bin · 64x2'
    assert entry['path'].startswith('puffer5:') and entry['path'].endswith(':64:2')

    spec = entry['path']
    assert gui.is_puffer5(spec)
    assert gui.parse_spec(spec) == (spec, None)     # not an MCTS spec
    assert gui.spec_label(spec) == entry['label']

    agent = gui.build_agent(spec, 2)
    assert isinstance(agent, PN.PufferNetAgent) and agent.name == entry['label']
    with pytest.raises(ValueError, match='4 seats'):
        gui.build_agent(spec, 4)
    with pytest.raises(ValueError, match='search needs'):
        gui.build_agent('mcts:' + spec, 2)
    with pytest.raises(ValueError, match='no such checkpoint'):
        gui.build_agent('puffer5:nope_weights.bin:64:2', 2)


@needs_fixture
def test_gui_checkpoint_scan_keeps_pt_files_working():
    """The .pt half of the scanner is unchanged, just tagged with a kind."""
    import gui

    for entry in gui.checkpoints():
        assert entry['kind'] in ('pt', 'puffer5')
        assert entry['num_players'] in (2, 3, 4)
        if entry['kind'] == 'pt':
            assert entry['path'].endswith('.pt')
