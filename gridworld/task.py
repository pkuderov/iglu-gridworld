from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
from gridworld.utils import int_2d, BUILD_ZONE_SIZE_X, BUILD_ZONE_SIZE_Z, BUILD_ZONE_SIZE


class Task:
    def __init__(self, target_grid: npt.NDArray[int], chat: str = '', last_instruction: str = None):
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
        # includes both the number of blocks to build and to remove
        self.n_target_diffs = np.count_nonzero(target_grid)

        self.chat = chat
        self.last_instruction = last_instruction

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


class TaskProgress:
    """Represents the progress of the task completion in the environment."""

    def __init__(
            self, task: Task, incremental: bool = True,
            initial_blocks: list[tuple] | npt.NDArray[int] = None,
            initial_grid: npt.NDArray[int] = None,
            full_grid: npt.NDArray[int] = None,
            invariant=True
    ):
        self.initial_blocks, self.initial_grid = resolve_blocks_grid(initial_blocks, initial_grid)
        self.incremental = incremental

        if self.incremental:
            # track only incremental progress, i.e. toward (target - initial) grid
            task = Task(target_grid=task.target_grid - self.initial_grid)
        self.task = task

        self.best_achieved_intersection = 0
        self.prev_n_grid_blocks = 0

        n_admissable_rotations = 4 if invariant else 1

        self.target_grids = [task.target_grid]
        # fill self.target_grids with four rotations of the original grid around the vertical axis
        for _ in range(n_admissable_rotations - 1):
            self.target_grids.append(_fill_grid_rotations(
                np.zeros(task.target_grid.shape, dtype=int),
                self.target_grids[-1]
            ))

        if full_grid is not None:
            self.full_grid = to_dense_grid(full_grid)
            self.n_full_diffs = np.count_nonzero(self.full_grid)

            full_grids = [self.full_grid]
            for _ in range(n_admissable_rotations - 1):
                full_grids.append(_fill_grid_rotations(
                    np.zeros(task.target_grid.shape, dtype=int),
                    full_grids[-1]
                ))
            compare_against_grids = full_grids
        else:
            self.full_grid = self.task.target_grid
            self.n_full_diffs = self.task.n_target_diffs
            compare_against_grids = self.target_grids

        if not invariant:
            self.admissible = [[(0, 0)]]
        else:
            self.admissible = [
                _get_admissible_points(compare_against_grids[i], self.n_full_diffs)
                for i in range(n_admissable_rotations)
            ]

    def reset(self):
        """
        placeholder method to have uniform interface with `Tasks` class.
        Resets all fields at initialization of the new episode.
        """

        self.best_achieved_intersection = 0
        if self.initial_blocks.size > 0 and not self.incremental:
            # initial blocks count toward the progress only for non-incremental tasks
            self.best_achieved_intersection = self.get_best_intersection(self.initial_grid)

        self.prev_n_grid_blocks = len(self.initial_blocks)

    def step_intersection(self, grid):
        """
        Calculates the difference between the maximal intersection at previous step and the current one.
        Note that the method updates object fields to save the grid size.

        Args (grid): current grid
        """
        best_intersection, n_grid_blocks, n_diffs = self.get_best_intersection(grid, lazy=True)

        n_improvements = best_intersection - self.best_achieved_intersection
        done = best_intersection == self.task.n_target_diffs

        self.prev_n_grid_blocks = n_grid_blocks
        self.best_achieved_intersection = best_intersection

        return n_improvements, n_diffs, done

    def get_best_intersection(self, grid, lazy: bool = False, with_full_stats: bool = False):
        if self.incremental:
            grid = grid - self.initial_grid

        n_grid_diffs = np.count_nonzero(grid)
        n_diffs_from_prev = self.prev_n_grid_blocks - n_grid_diffs

        if lazy and n_diffs_from_prev == 0:
            best_intersection, argmax = self.best_achieved_intersection, (0, 0, 0)
        else:
            best_intersection, argmax = self._get_best_intersection(grid)

        if not with_full_stats:
            return best_intersection, n_grid_diffs, n_diffs_from_prev

        n_target_diffs = self.task.n_target_diffs

        precision = safe_divide(best_intersection, n_grid_diffs)
        recall = safe_divide(best_intersection, n_target_diffs)
        f1 = safe_divide(2 * precision * recall, precision + recall)

        return dict(
            best_intersection=best_intersection,
            n_grid_blocks=n_grid_diffs,
            n_diffs=n_diffs_from_prev,
            argmax=argmax,
            precision=precision,
            recall=recall,
            f1=f1
        )

    def _get_best_intersection(self, grid):
        max_int, argmax = 0, (0, 0, 0)
        for i in range(len(self.admissible)):
            for dx, dz in self.admissible[i]:
                intersection = self.get_intersection(grid, dx, dz, i)
                if intersection > max_int:
                    max_int = intersection
                    argmax = (dx, dz, i)
        return max_int, argmax

    def get_intersection(self, grid, dx, dz, i):
        return _get_intersection(self.target_grids[i], grid, dx, dz)


def resolve_blocks_grid(
        blocks: list[tuple] | npt.NDArray[int] = None, grid: npt.NDArray[int] = None
):
    if grid is None:
        # induce from sparse blocks
        blocks = to_sparse_positions(blocks)
        grid = to_dense_grid(blocks)
        return blocks, grid

    # ensure format correctness
    grid = to_dense_grid(grid)
    blocks = to_sparse_positions(grid if blocks is None else blocks)
    return blocks, grid


@numba.jit(nopython=True, cache=True, inline='always')
def safe_divide(a, b) -> float:
    if b == 0:
        return 0.
    return a / b


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
            if np.count_nonzero(sls) == size:
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
    return np.count_nonzero((sls_target == sls_grid) & (sls_target != 0))


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
    is_dense_grid = isinstance(blocks, np.ndarray) and blocks.shape == BUILD_ZONE_SIZE
    if is_dense_grid:
        # already in correct dense format
        return blocks

    # let to_sparse_positions handle the format correctness
    blocks = to_sparse_positions(blocks)
    grid = np.zeros(BUILD_ZONE_SIZE, dtype=int)
    return _to_dense_grid(grid, blocks)


def to_sparse_positions(grid):
    if grid is None:
        grid = []

    if isinstance(grid, np.ndarray):
        if grid.shape[1] == 4:
            # already in correct sparse format
            return grid

        # TRANSFORMATION: dense -> sparse
        return _to_sparse_positions(grid)

    if isinstance(grid, (list, tuple)):
        # already in sparse format, but not as numpy array
        if len(grid) > 0:
            return np.array(grid, dtype=int)
        else:
            return np.empty((0, 4), dtype=int)

    raise ValueError(f'Invalid blocks type: {type(grid)} {grid}')


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
