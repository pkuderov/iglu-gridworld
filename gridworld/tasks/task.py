import numba
import numpy as np
import numpy.typing as npt
from gridworld.utils import int_2d, BUILD_ZONE_SIZE_X, BUILD_ZONE_SIZE_Z, BUILD_ZONE_SIZE


@numba.jit(nopython=True, cache=True)
def _fill_grid_rotations(grid: npt.NDArray[int], prev_grid: npt.NDArray[int]) -> npt.NDArray[int]:
    for x in range(BUILD_ZONE_SIZE_X):
        for z in range(BUILD_ZONE_SIZE_Z):
            grid[:, z, BUILD_ZONE_SIZE_X - x - 1] = prev_grid[:, x, z]
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


class Task:
    def __init__(
            self, chat, target_grid, last_instruction=None, starting_grid=None,
            full_grid=None, invariant=True
    ):
        """Creates a new Task represented with the past dialog and grid,
        the new instruction and target grid after completing the instruction.

        Parameters
        ----------
        chat : str
            Concatenation of all past utterances
        target_grid : numpy.array
            dense representation of the target
        last_instruction : string, optional
            the instruction corresponding to this step in the task,
            by default None
        starting_grid : numpy.array, optional
            sparse representation of the initial world, by default None
        full_grid : numpy.array, optional
            dense representation of the general target structure,
            by default None
        invariant : bool, optional
            by default True
        """

        assert target_grid.dtype == int

        self.chat = chat
        self.starting_grid = starting_grid
        self.last_instruction = last_instruction
        self.full_grid = full_grid
        self.target_size = (target_grid != 0).sum().item()
        self.full_size = self.target_size
        if full_grid is not None:
            self.full_size = (full_grid != 0).sum().item()
        self.target_grid = target_grid
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
        if self.starting_grid is not None:
            self.max_int = self.maximal_intersection(Tasks.to_dense(self.starting_grid))
        else:
            self.max_int = 0
        self.prev_grid_size = len(self.starting_grid) if self.starting_grid is not None else 0
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

    def argmax_intersection(self, grid):
        max_int, argmax = 0, (0, 0, 0)
        for i in range(len(self.admissible)):
            for dx, dz in self.admissible[i]:
                intersection = _get_intersection(self.target_grids[i], grid, dx, dz)
                if intersection > max_int:
                    max_int = intersection
                    argmax = (dx, dz, i)
        return argmax

    def maximal_intersection(self, grid):
        max_int = 0
        for i in range(len(self.admissible)):
            for dx, dz in self.admissible[i]:
                intersection = _get_intersection(self.target_grids[i], grid, dx, dz)
                if intersection > max_int:
                    max_int = intersection
        return max_int

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
def _get_intersection(target_grid, grid, dx, dz):
    x_sls = slice(max(dx, 0), BUILD_ZONE_SIZE_X + min(dx, 0))
    z_sls = slice(max(dz, 0), BUILD_ZONE_SIZE_Z + min(dz, 0))
    sls_target = target_grid[:, x_sls, z_sls]

    x_sls = slice(max(-dx, 0), BUILD_ZONE_SIZE_X + min(-dx, 0))
    z_sls = slice(max(-dz, 0), BUILD_ZONE_SIZE_Z + min(-dz, 0))
    sls_grid = grid[:, x_sls, z_sls]
    return ((sls_target == sls_grid) & (sls_target != 0)).sum()


@numba.jit(nopython=True, cache=True)
def _to_dense(grid, blocks):
    zone_x = BUILD_ZONE_SIZE_X // 2
    zone_z = BUILD_ZONE_SIZE_Z // 2
    for block in blocks:
        x, y, z, block_id = block
        grid[y + 1, x + zone_x, z + zone_z] = block_id
    return grid


def _to_sparse(blocks):
    zone_x = BUILD_ZONE_SIZE_X // 2
    zone_z = BUILD_ZONE_SIZE_Z // 2

    # TODO: it is not clear if the order of axes is correctly assumed for dense grid
    #   original: x, y, z = idx[0][i], idx[1][i], idx[2][i]
    #   possible: y, x, z = idx[0][i], idx[1][i], idx[2][i]
    #   the last one is exact inverse of the `to_dense` method, which looks more correct

    # yxz order in dense grid
    y, x, z = blocks.nonzero()
    colors = blocks[y, x, z]

    x -= zone_x
    y -= 1
    z -= zone_z

    n_blocks = len(x)
    new_blocks = np.empty((n_blocks, 4), dtype=int)
    new_blocks[:, 0] = x
    new_blocks[:, 1] = y
    new_blocks[:, 2] = z
    new_blocks[:, 3] = colors

    return new_blocks


class Tasks:
    """
    Represents many tasks where one can be active
    """
    @classmethod
    def to_dense(cls, blocks):
        is_list = isinstance(blocks, (list, tuple))
        is_arr = isinstance(blocks, np.ndarray) and blocks.shape[1] == 4

        if is_arr or is_list:
            grid = np.zeros(BUILD_ZONE_SIZE, dtype=int)
            if len(blocks) > 0:
                if is_list:
                    blocks = np.array(blocks, dtype=int)
                grid = _to_dense(grid, blocks)
            blocks = grid
        return blocks

    @classmethod
    def to_sparse(cls, blocks):
        if isinstance(blocks, np.ndarray) and blocks.shape[1] != 4:
            blocks = _to_sparse(blocks)
        return blocks

    def reset(self) -> Task:
        return NotImplemented

    def __len__(self) -> int:
        return NotImplemented

    def __iter__(self):
        return NotImplemented

    def set_task(self, task_id):
        return NotImplemented

    def get_target(self):
        return NotImplemented

    def set_task_obj(self, task: Task):
        return NotImplemented


class Subtasks(Tasks):
    """ Subtasks object represents a staged task where subtasks represent separate segments
    """
    def __init__(self, dialog, structure_seq, invariant=False, progressive=True) -> None:
        self.dialog = dialog
        self.invariant = invariant
        self.progressive = progressive
        self.structure_seq = structure_seq
        self.next = None
        self.full = False
        self.task_start = 0
        self.task_goal = 0
        self.full_structure = self.to_dense(self.structure_seq[-1])
        self.current = self.reset()

    def __getattr__(self, name):
        if name == 'current':
            return
        return getattr(self.current, name)

    def reset(self):
        """
        Randomly selects a random task within the task sequence.
        Each task is sampled with some non-trivial context (prior dialogs and
        starting structure) and one utterance goal instruction
        """
        if self.next is None:
            if len(self.structure_seq) == 1:
                turn = -1
            else:
                turn = np.random.choice(len(self.structure_seq)) - 1
            turn_goal = turn + 1
        else:
            turn = self.next
            turn_goal = self.next + 1
        self.task_start = turn
        self.task_goal = turn_goal
        self.current = self.create_task(self.task_start, self.task_goal)
        return self.current

    def __len__(self) -> int:
        return len(self.structure_seq)

    def __iter__(self):
        for i in range(len(self)):
            yield self.create_task(i - 1, i)

    def __repr__(self) -> str:
        return (f"Subtasks(total_steps={len(self.structure_seq)}, "
                f"current_task_start={self.task_start}, "
                f"current_task_end={self.task_goal})")

    def create_task(self, turn_start: int, turn_goal: int):
        """
        Returns a task with context defined by `turn_start` and goal defined
        by `turn_goal`

        """
        dialog = ''
        for turn in self.dialog[:turn_goal + 1]:
            if isinstance(turn, list):
                turn = '\n'.join(turn)
            dialog += '\n' + turn if len(dialog) > 0 else turn
        # dialog = '\n'.join([utt for utt in self.dialog[:turn_goal] if utt is not None])
        if turn_start == -1:
            initial_blocks = []
        else:
            initial_blocks = self.structure_seq[turn_start]
        tid = min(turn_goal, len(self.structure_seq) - 1) if not self.full else -1
        target_grid = self.structure_seq[tid]
        task = Task(
            dialog, target_grid=self.to_dense(target_grid),
            starting_grid=self.to_sparse(initial_blocks),
            full_grid=self.full_structure,
            last_instruction='\n'.join(self.dialog[tid])
        )
        # To properly init max_int and prev_grid_size fields
        task.reset()
        return task

    def step_intersection(self, grid):
        """

        """
        right_placement, wrong_placement, done = self.current.step_intersection(grid)
        if done and len(self.structure_seq) > self.task_goal and self.progressive:
            self.task_goal += 1
            self.current = self.create_task(self.task_start, self.task_goal)
            self.current.prev_grid_size = 0
            # to initialize things properly
            _, _, done = self.current.step_intersection(grid)
        return right_placement, wrong_placement, done

    def set_task(self, task_id):
        self.task_id = task_id
        self.current = self.create_task(task_id)
        return self.current

    def set_task_obj(self, task: Task):
        self.task_id = None
        self.current = task
        return self.current
