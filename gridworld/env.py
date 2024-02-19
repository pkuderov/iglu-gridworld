import gym
import numba
import numpy as np
from gridworld.task import Task, to_dense_grid, to_sparse_positions
from gridworld.utils import int_3d, BUILD_ZONE_SIZE
from gridworld.world import Agent, World
from gym import Env
from gym.spaces import Dict, Box, Discrete, Space


class String(Space):
    def __init__(self, ):
        super().__init__(shape=(), dtype=np.object_)

    def sample(self):
        return ''

    def contains(self, obj):
        return isinstance(obj, str)


# TODO:
#  - switch to gymnasium.Env API
#  - eliminate copying in the returned observation — ensure copying when needed on the user side


class GridWorld(Env):
    observation_space: Dict

    i_step: int
    max_steps: int

    def __init__(
            self, max_steps=250,
            action_space='walking', discretize=False, vector_state=True,
            render=True, render_size=(64, 64), target_in_obs=False,
            select_and_place=False,
            compute_reward=True, right_placement_scale=1., wrong_placement_scale=0.1,
            fake=False
    ):
        is_flying = action_space == 'flying'
        self.agent = Agent(is_flying=is_flying)
        self.world = World()
        self.world.add_callback('on_add', self._add_block)
        self.world.add_callback('on_remove', self._remove_block)

        # TODO: move to world
        self.grid = np.zeros(BUILD_ZONE_SIZE, dtype=int)
        self.initial_blocks = to_sparse_positions([])
        self.initial_position = (0, 0, 0)
        self.initial_rotation = (0, 0)

        self.i_step = 0
        self.max_steps = max_steps

        self.task = None
        self.require_reset = True

        self.compute_reward = compute_reward
        self.right_placement_scale = right_placement_scale
        self.wrong_placement_scale = wrong_placement_scale
        self.right_placement = 0
        self.wrong_placement = 0

        self.render_size = render_size
        self.select_and_place = select_and_place
        self.target_in_obs = target_in_obs
        self.vector_state = vector_state
        self.discretize = discretize
        self.action_space_type = action_space
        self.fake = fake

        self._overwrite_initial_blocks = None
        self._diff_init_grid = None
        self._diff_task = None

        self.action_space = get_action_space(action_space, discretize)
        self.observation_space = get_observation_space(
            render, target_in_obs, vector_state, self.render_size
        )

        self.do_render = render
        self.renderer = self._setup_pyglet_renderer() if render and not fake else None
        self.world.initialize()

    def set_task(self, task: Task):
        """
        Assigns provided task into the environment. On each .reset, the env
        Queries the .reset method for the task object. This method should drop
        the task state to the initial one.
        Note that the env can only work with non-None task or task generator.
        """
        self.task = task
        self.require_reset = True

    def initialize_world(self, initial_blocks, initial_position):
        self._overwrite_initial_blocks = initial_blocks
        self.initial_position = tuple(initial_position[:3])
        self.initial_rotation = tuple(initial_position[3:])
        self.require_reset = True

    def reset(self, **_):
        self.i_step = 0

        if self._overwrite_initial_blocks is not None:
            self.initial_blocks = self._overwrite_initial_blocks
        else:
            self.initial_blocks = self.task.initial_blocks

        for block in set(self.world.placed):
            self.world.remove_block(block)
        for x, y, z, color in self.initial_blocks:
            self.world.add_block((x, y, z), color)

        self.agent.position = self.initial_position
        self.agent.rotation = self.initial_rotation
        self.agent.inventory = np.full(6, 20, dtype=int)
        self.agent.inventory[self.initial_blocks[:, -1] - 1] -= 1

        self.task.reset()
        if self.initial_blocks.size > 0:
            # make diff task only if there are initial blocks
            self._diff_init_grid = to_dense_grid(self.initial_blocks)
            self._diff_task = Task(
                # create a synthetic task with only diff blocks.
                # blocks to remove have negative ids.
                target_grid=self.task.target_grid - self._diff_init_grid
            )
            self._diff_task.reset()
        else:
            self._diff_task = None
            self._diff_init_grid = None

        obs = {
            'inventory': self.agent.inventory,
            'compass': np.array([0.], dtype=float),
            'dialog': self.task.chat
        }
        if self.vector_state:
            obs['grid'] = self.grid.copy()
            obs['agentPos'] = np.array([0., 0., 0., 0., 0.], dtype=float)
        if self.target_in_obs:
            obs['target_grid'] = self.task.target_grid.copy()
        if self.do_render:
            obs['pov'] = self.render()

        self.require_reset = False
        return obs

    def step(self, action):
        assert not self.require_reset, 'Environment is not reset'
        self.i_step += 1

        self.world.step(
            self.agent, action, select_and_place=self.select_and_place,
            action_space=self.action_space_type, discretize=self.discretize
        )

        x, y, z = self.agent.position
        yaw, pitch = self.agent.rotation

        obs = {
            'inventory': self.agent.inventory,
            'compass': np.array([yaw - 180., ], dtype=float),
            'dialog': self.task.chat
        }

        if self.vector_state:
            obs['grid'] = self.grid.copy()
            obs['agentPos'] = np.array([x, y, z, pitch, yaw], dtype=float)

        terminated, reward = self.calculate_progress()
        truncated = self.i_step == self.max_steps
        done = terminated or truncated

        if self.target_in_obs:
            obs['target_grid'] = self.task.target_grid.copy()
        if self.do_render:
            obs['pov'] = self.render()
        return obs, reward, done, {}

    def render(self, *_, **__):
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
        if self._diff_task is not None:
            diff_grid = self.grid - self._diff_init_grid
            n_correct, n_incorrect, done = self._diff_task.step_intersection(diff_grid)
        else:
            n_correct, n_incorrect, done = self.task.step_intersection(self.grid)

        if n_correct == 0:
            reward = n_incorrect * self.wrong_placement_scale
        else:
            reward = n_correct * self.right_placement_scale
        return done, reward

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
            dir_path=dir_path, caption='Pyglet', resizable=False,
        )
        # setup callbacks
        self.world.add_callback('on_add', renderer.add_block)
        self.world.add_callback('on_remove', renderer.remove_block)

        return renderer


def get_action_space(action_space, discretize):
    assert action_space in ('walking', 'flying'), (
        f'Unknown action space: {action_space}'
    )

    if action_space == 'flying':
        return Dict({
            'movement': Box(low=-1, high=1, shape=(3,), dtype=float),
            'camera': Box(low=-5, high=5, shape=(2,), dtype=float),
            'inventory': Discrete(7),
            'placement': Discrete(3),
        })

    # walking
    if discretize:
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
