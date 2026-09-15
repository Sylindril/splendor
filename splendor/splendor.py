"""PufferEnv wrapper around the C Splendor game (splendor/splendor.h)."""
import numpy as np
import gymnasium

import pufferlib
from splendor import binding
from splendor import layout


class Splendor(pufferlib.PufferEnv):
    """`num_envs` independent games of `num_players` seats each.

    One PufferLib agent per seat: game i owns agent rows [i*P, (i+1)*P).
    One env step = one Splendor turn; only the current seat's action is used.
    max_turns counts total turns (all seats); None means 60 per seat.
    auto_reset=False (for GUIs) keeps a finished board, with an all-zero legal
    mask, until reset() is called; training always uses auto_reset=True.
    """

    def __init__(self, num_envs=1024, num_players=2, max_turns=None,
                 reward_point=0.02, reward_card=0.0, reward_win=1.0,
                 reward_loss=-1.0, report_interval=128, render_mode=None,
                 auto_reset=True, buf=None, seed=0):
        if not 2 <= num_players <= 4:
            raise ValueError('num_players must be 2, 3 or 4')
        if max_turns is None:
            max_turns = 60 * num_players  # total turns, i.e. 60 per seat

        self.num_envs = num_envs
        self.num_players = num_players
        self.max_turns = max_turns
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.obs_n = layout.obs_size(num_players)

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_n,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(layout.NUM_ACTIONS)
        self.num_agents = num_envs * num_players

        super().__init__(buf=buf)

        p = num_players
        handles = []
        for i in range(num_envs):
            handles.append(binding.env_init(
                self.observations[i*p:(i+1)*p],
                self.actions[i*p:(i+1)*p],
                self.rewards[i*p:(i+1)*p],
                self.terminals[i*p:(i+1)*p],
                self.truncations[i*p:(i+1)*p],
                i + seed*num_envs,
                num_players=num_players,
                max_turns=max_turns,
                reward_point=reward_point,
                reward_card=reward_card,
                reward_win=reward_win,
                reward_loss=reward_loss,
                auto_reset=int(bool(auto_reset)),
            ))

        self.handles = handles
        self.c_envs = binding.vectorize(*handles)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, 0 if seed is None else int(seed))
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        self.tick += 1

        info = []
        if self.tick % self.report_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)

        return (self.observations, self.rewards,
                self.terminals, self.truncations, info)

    def get_state(self, env_index):
        """Opaque snapshot (bytes) of one game; restore with put_state(i, state=...)."""
        return binding.env_get(self.handles[env_index])['state']

    def put_state(self, env_index, **kwargs):
        """put_state(i, state=bytes) restores a snapshot; put_state(i, bonuses=[P*5
        ints]) overwrites bonus cards (for tests); put_state(i, determinize=seat)
        re-deals everything `seat` cannot see. Observations are refreshed.

        Stepping a game with action NOOP (-1) for its acting seat leaves it
        untouched (used to freeze games while other agents think)."""
        binding.env_put(self.handles[env_index], **kwargs)

    def render(self, env_index=0):
        binding.vec_render(self.c_envs, env_index)

    def close(self):
        binding.vec_close(self.c_envs)


def test_performance(num_envs=1024, num_players=2, timeout=5.0, legal=True):
    """SPS benchmark. `legal` samples uniformly from the action mask (real
    games, but pays numpy overhead); otherwise uses cached random actions
    (invalid ones count as PASS) to measure raw C throughput."""
    import time
    env = Splendor(num_envs=num_envs, num_players=num_players,
                   report_interval=10**9)
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    cache = rng.integers(0, layout.NUM_ACTIONS,
                         (1024, env.num_agents)).astype(np.int32)

    env.step(cache[0])  # warm up
    tick = 0
    start = time.time()
    while time.time() - start < timeout:
        if legal:
            actions = layout.random_legal_actions(obs, num_players, rng)
        else:
            actions = cache[tick % 1024]
        obs = env.step(actions)[0]
        tick += 1
    elapsed = time.time() - start

    kind = 'random legal' if legal else 'random raw'
    print(f'{kind:<13} envs={num_envs} players={num_players} steps={tick}')
    print(f'  turns/s:       {num_envs*tick/elapsed:,.0f}')
    print(f'  agent-steps/s: {env.num_agents*tick/elapsed:,.0f}')
    env.close()


if __name__ == '__main__':
    test_performance(legal=False)
    test_performance(legal=True)
