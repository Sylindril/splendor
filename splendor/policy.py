"""Masked MLP policies for Splendor. Imports no compiled extension."""
import torch
from torch import nn

import pufferlib.models
from pufferlib.pytorch import layer_init

from splendor import layout


class Policy(nn.Module):
    """GELU MLP over the scaled observation, with action masking.

    The legal-action mask lives inside the observation; it is stashed during
    encode_observations so that decode_actions can apply it after the
    (optional) LSTM. `layers` hidden layers of width `hidden_size`, each
    optionally followed by LayerNorm.
    """

    def __init__(self, env, hidden_size=256, layers=2, norm=False):
        super().__init__()
        obs_n = int(env.single_observation_space.shape[0])
        num_players = layout.num_players_from_obs(obs_n)

        self.hidden_size = hidden_size
        self.is_continuous = False
        self.mask_start = layout.mask_offset(num_players)
        self.mask_end = layout.turn_offset(num_players)
        self.register_buffer('scale',
            torch.from_numpy(layout.obs_scale(num_players)), persistent=False)

        blocks, width = [], obs_n
        for _ in range(layers):
            blocks.append(layer_init(nn.Linear(width, hidden_size)))
            if norm:
                blocks.append(nn.LayerNorm(hidden_size))
            blocks.append(nn.GELU())
            width = hidden_size
        self.encoder = nn.Sequential(*blocks)
        self.actor = layer_init(
            nn.Linear(hidden_size, layout.NUM_ACTIONS), std=0.01)
        self.value_fn = layer_init(nn.Linear(hidden_size, 1), std=1)
        self._mask = None

    def encode_observations(self, observations, state=None):
        obs = observations.view(observations.shape[0], -1).float()
        self._mask = obs[:, self.mask_start:self.mask_end] > 0
        return self.encoder(obs * self.scale)

    def decode_actions(self, hidden):
        logits = self.actor(hidden)
        # -1e8 rather than -inf so entropy and logsumexp stay finite
        logits = torch.where(self._mask, logits, torch.full_like(logits, -1e8))
        return logits, self.value_fn(hidden)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        return self.decode_actions(hidden)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)


class Big(Policy):
    """3 x 512 LayerNorm MLP: the default for serious runs."""

    def __init__(self, env, hidden_size=512, layers=3, norm=True):
        super().__init__(env, hidden_size, layers, norm)


class Recurrent(pufferlib.models.LSTMWrapper):
    def __init__(self, env, policy, input_size=None, hidden_size=None):
        input_size = input_size or policy.hidden_size
        hidden_size = hidden_size or policy.hidden_size
        super().__init__(env, policy, input_size, hidden_size)


def arch_from_state_dict(sd):
    """(hidden_size, layers, norm) of a Policy/Big checkpoint, from its keys."""
    linears = sorted(int(k.split('.')[1]) for k in sd
                     if k.startswith('encoder.') and k.endswith('.weight')
                     and sd[k].dim() == 2)
    norm = any(k.startswith('encoder.') and sd[k].dim() == 1
               and k.endswith('.weight') for k in sd)
    hidden = int(sd['actor.weight'].shape[1])
    return hidden, len(linears), norm
