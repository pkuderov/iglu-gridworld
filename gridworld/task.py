from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
from gridworld.utils import int_2d, BUILD_ZONE_SIZE_X, BUILD_ZONE_SIZE_Z, BUILD_ZONE_SIZE


class Task:
    def __init__(
            self,
            target_grid: npt.NDArray[int],
            initial_blocks: list[tuple] | npt.NDArray[int] = None,
            chat: str = '', last_instruction: str = None,
            full_grid=None, invariant=True
    ):
        """Creates a new Task represented with the past dialog and grid,
        the new instruction and target grid after completing the instruction.

        Chat — concatenation of all past utterances.
        last_instruction : string, optional
            the instruction corresponding to this step in the task,
            by default None
        initial_blocks : numpy.array, optional
            sparse representation of the initial world, by default None
        full_grid : numpy.array, optional
            dense representation of the general target structure,
            by default None
        """

        assert target_grid.dtype == int

        self.target_grid = to_dense_grid(target_grid)
        self.target_size = np.count_nonzero(target_grid)

        if initial_blocks is None:
            initial_blocks = []
        self.initial_blocks = to_sparse_positions(initial_blocks)

        self.chat = chat
        self.last_instruction = last_instruction

        self.full_grid = full_grid
        self.full_size = np.count_nonzero(full_grid) if full_grid is not None else self.target_size

        self.max_int = 0
        self.prev_grid_size = 0
        self.right_placement = 0
        self.wrong_placement = 0

        self.target_grids = [target_grid]
        # fill self.target_grids with four rotations of the original grid around the vertical axis
        for _ in range(3):
            self.target_grids.append(_fill_grid_rotations(
                np.zeros(target_grid.shape, dtype=int), self.target_grids[-1]
            ))

        full_grids = None
        if full_grid is not None:
            full_grids = [full_grid]
            for _ in range(3):
                full_grids.append(_fill_grid_rotations(
                    np.zeros(target_grid.shape, dtype=int), full_grids[-1]
                ))

        if not invariant:
            self.admissible = [[(0, 0)]]
        else:
            self.admissible = [
                _get_admissible_points(
                    full_grids[i] if full_grids is not None else self.target_grids[i],
                    self.full_size
                )
                for i in range(4)
            ]

    def reset(self):
        """
        placeholder method to have uniform interface with `Tasks` class.
        Resets all fields at initialization of the new episode.
        """
        if self.initial_blocks is not None:
            self.max_int = self.maximal_intersection(
                to_dense_grid(self.initial_blocks)
            )
        else:
            self.max_int = 0
        self.prev_grid_size = len(self.initial_blocks) if self.initial_blocks is not None else 0
        self.right_placement = 0
        self.wrong_placement = 0
        return self

    def step_intersection(self, grid):
        """
        Calculates the difference between the maximal intersection at previous step and the current one.
        Note that the method updates object fields to save the grid size.

        Args (grid): current grid
        """
        grid_size = (grid != 0).sum().item()
        wrong_placement = (self.prev_grid_size - grid_size)
        max_int = self.maximal_intersection(grid) if wrong_placement != 0 else self.max_int
        right_placement = (max_int - self.max_int)
        done = max_int == self.target_size

        self.prev_grid_size = grid_size
        self.max_int = max_int
        self.right_placement = right_placement
        self.wrong_placement = wrong_placement
        return right_placement, wrong_placement, done

    def maximal_intersection(self, grid, with_argmax=False):
        max_int, argmax = 0, (0, 0, 0)
        for i in range(len(self.admissible)):
            for dx, dz in self.admissible[i]:
                intersection = self.get_intersection(grid, dx, dz, i)
                if intersection > max_int:
                    max_int = intersection
                    argmax = (dx, dz, i)

        if with_argmax:
            return max_int, argmax
        return max_int

    def get_intersection(self, grid, dx, dz, i):
        return _get_intersection(self.target_grids[i], grid, dx, dz)

    # placeholder methods for uniformity with Tasks class
    ###
    def __len__(self):
        return 1

    def __iter__(self):
        yield self
    ###

    def __repr__(self) -> str:
        instruction = self.last_instruction \
            if len(self.last_instruction) < 20 \
            else self.last_instruction[:20] + '...'
        return f"Task(instruction={instruction})"


@numba.jit(nopython=True, cache=True)
def _fill_grid_rotations(grid: npt.NDArray[int], prev_grid: npt.NDArray[int]) -> npt.NDArray[int]:
    for x in range(BUILD_ZONE_SIZE_X):
        grid[:, :, BUILD_ZONE_SIZE_X - 1 - x] = prev_grid[:, x, :]
    return grid


@numba.jit(nopython=True, cache=True)
def _get_admissible_points(grid: npt.NDArray[int], size: int) -> list[int_2d]:
    # (dx, dz) is admissible iff the translation of target grid by (dx, dz)
    # preserve (== doesn't cut) target structure within original (not shifted) target grid
    admissible = []
    for dx in range(-BUILD_ZONE_SIZE_X + 1, BUILD_ZONE_SIZE_X):
        low_dx = max(dx, 0)
        high_dx = BUILD_ZONE_SIZE_X + min(dx, 0)

        for dz in range(-BUILD_ZONE_SIZE_Z + 1, BUILD_ZONE_SIZE_Z):
            low_dz = max(dz, 0)
            high_dz = BUILD_ZONE_SIZE_Z + min(dz, 0)
            sls = grid[:, low_dx:high_dx, low_dz:high_dz]
            if np.sum(sls != 0) == size:
                admissible.append((dx, dz))
    return admissible


@numba.jit(nopython=True, cache=True)
def _get_intersection(target_grid, grid, dx, dz):
    x_sls = slice(max(dx, 0), BUILD_ZONE_SIZE_X + min(dx, 0))
    z_sls = slice(max(dz, 0), BUILD_ZONE_SIZE_Z + min(dz, 0))
    sls_target = target_grid[:, x_sls, z_sls]

    x_sls = slice(max(-dx, 0), BUILD_ZONE_SIZE_X + min(-dx, 0))
    z_sls = slice(max(-dz, 0), BUILD_ZONE_SIZE_Z + min(-dz, 0))
    sls_grid = grid[:, x_sls, z_sls]
    return ((sls_target == sls_grid) & (sls_target != 0)).sum()


@numba.jit(nopython=True, cache=True)
def _to_dense_grid(grid, blocks):
    for block in blocks:
        x, y, z, block_id = block
        grid[y + 1, x + 5, z + 5] = block_id
    return grid


def _to_sparse_positions(blocks):
    # TODO: it is not clear if the order of axes is correctly assumed for dense grid
    #   original: x, y, z = idx[0][i], idx[1][i], idx[2][i]
    #   possible: y, x, z = idx[0][i], idx[1][i], idx[2][i]
    #   the last one is exact inverse of the `to_dense` method, which looks more correct

    # yxz order in dense grid
    y, x, z = blocks.nonzero()
    colors = blocks[y, x, z]

    y -= 1
    x -= 5
    z -= 5

    n_blocks = len(x)
    positions = np.empty((n_blocks, 4), dtype=int)
    positions[:, 0] = x
    positions[:, 1] = y
    positions[:, 2] = z
    positions[:, 3] = colors

    return positions


def to_dense_grid(blocks):
    is_sparse_list = isinstance(blocks, (list, tuple))
    is_sparse_np = isinstance(blocks, np.ndarray) and blocks.shape[1] == 4

    if is_sparse_np or is_sparse_list:
        blocks = to_sparse_positions(blocks)
        grid = np.zeros(BUILD_ZONE_SIZE, dtype=int)
        return _to_dense_grid(grid, blocks)

    return blocks


def to_sparse_positions(blocks):
    if isinstance(blocks, np.ndarray):
        if blocks.shape[1] == 4:
            # already in correct sparse format
            return blocks
        return _to_sparse_positions(blocks)

    if isinstance(blocks, (list, tuple)):
        # already in sparse format, but not as numpy array
        if len(blocks) > 0:
            return np.array(blocks, dtype=int)
        else:
            return np.empty((0, 4), dtype=int)

    raise ValueError(f'Invalid blocks type: {type(blocks)} {blocks}')


class Tasks:
    """Represents many tasks where one can be active"""
    def reset(self) -> Task:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def set_task(self, task_id):
        raise NotImplementedError()

    def get_target(self):
        raise NotImplementedError()

    def set_task_obj(self, task: Task):
        raise NotImplementedError()
