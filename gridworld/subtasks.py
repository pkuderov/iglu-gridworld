from __future__ import annotations

import numpy as np
from gridworld.task import Tasks, Task, to_dense_grid, to_sparse_positions


class Subtasks(Tasks):
    """ Subtasks object represents a staged task where subtasks represent separate segments
    """
    def __init__(
            self, dialog, structure_seq, invariant=False, progressive=True
    ):
        self.dialog = dialog
        self.invariant = invariant
        self.progressive = progressive
        self.structure_seq = structure_seq
        self.next = None
        self.full = False
        self.task_start = 0
        self.task_goal = 0
        self.full_structure = to_dense_grid(self.structure_seq[-1])
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
            target_grid=to_dense_grid(target_grid),
            initial_blocks=to_sparse_positions(initial_blocks),
            full_grid=self.full_structure,
            chat=dialog, last_instruction='\n'.join(self.dialog[tid])
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
