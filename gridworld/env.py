from __future__ import annotations

from typing import Any

import gymnasium as gym
import numba
import numpy as np
from gridworld.task import Task, to_sparse_positions, TaskProgress
from gridworld.utils import int_3d, BUILD_ZONE_SIZE
from gridworld.world import Agent, World
from gymnasium.core import ObsType, RenderFrame


class String(gym.Space):
    def __init__(self, ):
        super().__init__(shape=(), dtype=np.object_)

    def sample(self, mask: Any | None = None):
        return ''

    def contains(self, obj):
        return isinstance(obj, str)


# TODO:
#  - switch to gymnasium.Env API
#  - eliminate copying in the returned observation — ensure copying when needed on the user side
#  - switch String space to already existing Text space


class GridWorld(gym.Env):
    observation_space: gym.spaces.Dict

    i_step: int
    max_steps: int

    def __init__(
            self, max_steps=250,
            action_space='walking', discretize=False, vector_state=True,
            render=True, render_size=(64, 64), target_in_obs=False,
            select_and_place=False, right_placement_scale=1., wrong_placement_scale=0.1,
            fake=False, track_progress: bool = False, task_progress_params: dict = None,
            renderer_invert_y=False
    ):
        is_flying = action_space == 'flying'
        self.agent = Agent(is_flying=is_flying)
        self.world = World(flying=is_flying, discrete_actions=discretize)
        self.world.add_callback('on_add', self._add_block)
        self.world.add_callback('on_remove', self._remove_block)

        # TODO: move to world
        self.grid = np.zeros(BUILD_ZONE_SIZE, dtype=int)
        self.initial_blocks = to_sparse_positions([])
        self.initial_position = (0., 0., 0.)
        self.initial_rotation = (0., 0.)

        self.i_step = 0
        self.max_steps = max_steps

        self.task = None
        self.require_reset = True

        self.track_progress = track_progress
        if task_progress_params is None:
            # track progress initialization params
            task_progress_params = {}
        self.track_progress_params = task_progress_params
        self.task_progress = None
        self.right_placement_scale = right_placement_scale
        self.wrong_placement_scale = wrong_placement_scale

        self.render_size = render_size
        self.select_and_place = select_and_place
        self.target_in_obs = target_in_obs
        self.vector_state = vector_state
        self.discrete_actions = discretize
        self.action_space_type = action_space
        self.fake = fake

        self.action_space = get_action_space(action_space, self.discrete_actions)
        self.observation_space = get_observation_space(
            render, target_in_obs, vector_state, self.render_size
        )

        self.do_render = render
        self.renderer_invert_y = renderer_invert_y
        self.renderer = self._setup_pyglet_renderer() if render and not fake else None
        self.world.initialize()

    def set_task(self, task: Task):
        """Assigns provided task into the environment."""
        self.task = task
        self.require_reset = True

    def set_world_initial_state(self, initial_blocks=None, initial_agent_state=None):
        """Sets initial state of the world: initial blocks and agent position/rotation."""

        if initial_blocks is not None:
            self.initial_blocks = to_sparse_positions(initial_blocks)

        if initial_agent_state is not None:
            x, y, z, yaw, pitch = initial_agent_state
            self.initial_position = (x, y, z)
            self.initial_rotation = (yaw, pitch)

        self.require_reset = initial_blocks is not None or initial_agent_state is not None

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)

        self.i_step = 0

        for block in set(self.world.placed):
            self.world.remove_block(block)
        for x, y, z, color in self.initial_blocks:
            self.world.add_block((x, y, z), color)

        self.agent.position = self.initial_position
        self.agent.rotation = self.initial_rotation
        self.agent.inventory = np.full(6, 20, dtype=int)
        self.agent.inventory[self.initial_blocks[:, -1] - 1] -= 1

        if self.track_progress:
            self.task_progress = self.create_progress_tracker()

        obs = self.observation()
        info = {}
        self.require_reset = False
        return obs, info

    def step(self, action):
        assert not self.require_reset, 'Environment is not reset'
        self.i_step += 1

        self.world.step(self.agent, action, select_and_place=self.select_and_place)

        obs = self.observation()
        terminated, reward = self.calculate_progress()
        truncated = self.i_step >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info

    def observation(self) -> ObsType:
        x, y, z = self.agent.position
        yaw, pitch = self.agent.rotation

        obs = {
            'inventory': self.agent.inventory,
            'compass': np.array([yaw - 180.], dtype=float),
            'dialog': self.task.chat
        }
        if self.vector_state:
            obs['grid'] = self.grid.copy()
            obs['agentPos'] = np.array([x, y, z, pitch, yaw], dtype=float)
        if self.target_in_obs:
            obs['target_grid'] = self.task.target_grid.copy()
        if self.do_render:
            obs['pov'] = self.render()

        return obs

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if not self.do_render:
            raise ValueError('create env with render=True')

        if self.fake:
            return self.observation_space['pov'].sample()

        return self.renderer.render(
            self.agent.position, self.agent.rotation
        )

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            del self.renderer

        super().close()

    def calculate_progress(self):
        if self.track_progress:
            n_correct, n_incorrect, done = self.task_progress.step_intersection(self.grid)
        else:
            n_correct, n_incorrect, done = 0, 0, False

        if n_correct == 0:
            reward = n_incorrect * self.wrong_placement_scale
        else:
            reward = n_correct * self.right_placement_scale
        return done, reward

    def create_progress_tracker(self):
        return TaskProgress(
            task=self.task, initial_blocks=self.initial_blocks,
            **self.track_progress_params
        )

    def is_flying(self):
        return self.agent.flying

    def _add_block(self, position, kind, build_zone=True):
        if self.world.initialized and build_zone:
            x, y, z = to_grid_space(position)
            self.grid[y, x, z] = kind

    def _remove_block(self, position, build_zone=True):
        if self.world.initialized and build_zone:
            x, y, z = to_grid_space(position)
            if self.grid[y, x, z] == 0:
                raise ValueError(
                    f'Removal of non-existing block. address: y={y}, x={x}, z={z}; '
                    f'grid state: {self.grid.nonzero()[0]};'
                )
            self.grid[y, x, z] = 0

    def _setup_pyglet_renderer(self):
        import os
        import gridworld
        from gridworld.render import Renderer

        dir_path = os.path.dirname(gridworld.__file__)
        width, height = self.render_size
        renderer = Renderer(
            width=width, height=height,
            dir_path=dir_path, invert_y=self.renderer_invert_y,
            caption='Pyglet', resizable=False,
        )
        # setup callbacks
        self.world.add_callback('on_add', renderer.add_block)
        self.world.add_callback('on_remove', renderer.remove_block)

        return renderer


def get_action_space(action_space, discretize):
    assert action_space in ('walking', 'flying'), (
        f'Unknown action space: {action_space}'
    )
    from gymnasium.spaces import Dict, Box, Discrete

    # flying continuous
    if action_space == 'flying':
        return Dict({
            'movement': Box(low=-1, high=1, shape=(3,), dtype=float),
            'camera': Box(low=-5, high=5, shape=(2,), dtype=float),
            'inventory': Discrete(7),
            'placement': Discrete(3),
        })

    # walking discrete
    if discretize:
        # 0 noop; 1 forward; 2 back; 3 left; 4 right; 5 jump; 6-11 hotbar;
        # 12 camera left; 13 camera right; 14 camera up; 15 camera down;
        # 16 attack; 17 use;
        return Discrete(18)

    # walking continuous
    return Dict({
        'forward': Discrete(2),
        'back': Discrete(2),
        'left': Discrete(2),
        'right': Discrete(2),
        'jump': Discrete(2),
        'attack': Discrete(2),
        'use': Discrete(2),
        'camera': Box(low=-5, high=5, shape=(2,)),
        'hotbar': Discrete(7)
    })


def get_observation_space(
        render, target_in_obs, vector_state, render_size
):
    from gymnasium.spaces import Box, Dict

    observation_space = {
        'inventory': Box(low=0, high=20, shape=(6,), dtype=int),
        'compass': Box(low=-180, high=180, shape=(1,), dtype=float),
        'dialog': String()
    }

    if vector_state:
        observation_space['agentPos'] = Box(
            low=np.array([-8, -2, -8, -90, 0], dtype=float),
            high=np.array([8, 12, 8, 90, 360], dtype=float),
            shape=(5,)
        )
        observation_space['grid'] = Box(
            low=-1, high=7, shape=BUILD_ZONE_SIZE, dtype=int
        )

    if target_in_obs:
        observation_space['target_grid'] = Box(
            low=-1, high=7, shape=BUILD_ZONE_SIZE, dtype=int
        )

    if render:
        width, height = render_size
        observation_space['pov'] = Box(
            low=0, high=255, shape=(width, height, 3), dtype=np.uint8
        )
    return Dict(observation_space)


@numba.jit(nopython=True, cache=True, inline='always')
def to_grid_space(pos_3d: int_3d) -> int_3d:
    x, y, z = pos_3d
    return x + 5, y + 1, z + 5


gym.envs.register(
     id='IGLUGridworld-v0',
     entry_point='gridworld.env:create_env',
     kwargs={}
)

gym.envs.register(
     id='IGLUGridworldVector-v0',
     entry_point='gridworld.env:create_env',
     kwargs={'vector_state': True, 'render': False}
)
