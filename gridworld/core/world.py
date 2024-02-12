import math
from typing import Optional

import numba
from numba.typed.typeddict import Dict

from ..utils import (
    WHITE, GREY, BLUE, FACES, int_3d, FLYING_SPEED, WALKING_SPEED, GRAVITY,
    TERMINAL_VELOCITY, PLAYER_HEIGHT, JUMP_SPEED, discretize_3d, float_3d, float_2d, int_2d,
    to_float_3d, to_int_3d
)


PLAYER_PAD = 0.25


class Agent:
    __slots__ = (
        'flying', 'strafe', 'position', 'rotation', 'reticle', 'sustain',
        'dy', 'time_int_steps', 'inventory', 'active_block'
    )

    position: float_3d

    def __init__(self, sustain=False) -> None:
        # When flying gravity has no effect and speed is increased.
        self.flying = False
        self.strafe = (0, 0)
        self.position = (0., 0., 0.)
        self.rotation = (0., 0.)
        self.reticle = None

        # actions are long-lasting state switches
        self.sustain = sustain

        # Velocity in the y (upward) direction.
        self.dy = 0
        self.time_int_steps = 2
        self.inventory = [
            20, 20, 20, 20, 20, 20
        ]
        self.active_block = BLUE


class World:
    __slots__ = (
        'world', 'init_blocks', 'shown', 'placed', 'callbacks', 'initialized'
    )

    world: dict[int_3d, int]
    # a set of block coordinates along with their colors that are placed initially
    init_blocks: dict[int_3d, int]

    def __init__(self):
        # make it
        self.world = Dict.empty(
            key_type=numba.typeof((0, 0, 0)),
            value_type=numba.typeof(0)
        )
        self.init_blocks = Dict.empty(
            key_type=numba.typeof((0, 0, 0)),
            value_type=numba.typeof(0)
        )
        _add_initial_blocks(self.init_blocks)

        self.shown = {}
        self.placed = set()
        self.callbacks = {
            'on_add': [],
            'on_remove': []
        }
        self.initialized = False

    def add_callback(self, name, func):
        self.callbacks[name].append(func)

    # ========= BLOCKS RELATED METHODS =========
    def reset(self):
        for position in self.placed:
            self.remove_block(position)
        self.initialized = False
        for position in self.world:
            self.remove_block(position)

        assert len(self.world) == 0
        self.world.clear()
        self.shown = {}
        self.placed = set()

    def initialize(self):
        for position, texture in self.init_blocks.items():
            self.add_block(position, texture)
        self.initialized = True

    def hit_test(self, position, vector, max_distance=8):
        """
        Line of sight search from current position. If a block is
        intersected it is returned, along with the block previously in the line
        of sight. If no block is found, return None, None.

        Parameters
        ----------
        position : tuple of len 3
            The (x, y, z) position to check visibility from.
        vector : tuple of len 3
            The line of sight vector.
        max_distance : int
            How many blocks away to search for a hit.

        """
        position = to_float_3d(position)
        return _hit_test(self.world, position, vector, max_distance)

    def add_block(self, block_position: int_3d, texture: int):
        """ Add a block with the given `texture` and `position` to the world.

        Parameters
        ----------
        block_position : tuple of len 3
            The (x, y, z) position of the block to add.
        texture : list of len 3
            The coordinates of the texture squares. Use `texture_coordinates()` to
            generate.
        """
        block_position = to_int_3d(block_position)
        x, y, z = block_position

        if block_position in self.world:
            self.remove_block(block_position)

        self.world[block_position] = texture
        self.shown[block_position] = texture

        build_zone = _is_build_zone(x, y, z)
        for cb in self.callbacks['on_add']:
            cb(block_position, texture, build_zone=build_zone)
        if self.initialized:
            self.placed.add(block_position)

    def remove_block(self, block_position: int_3d):
        """ Remove the block at the given `position."""
        block_position = to_int_3d(block_position)
        x, y, z = block_position

        self.world.pop(block_position)

        if block_position in self.shown:
            self.shown.pop(block_position)
            build_zone = _is_build_zone(x, y, z)
            for cb in self.callbacks['on_remove']:
                cb(block_position, build_zone=build_zone)

        if self.initialized:
            self.placed.remove(block_position)
    # ========= END BLOCKS RELATED METHODS =========

    # ========= AGENT CONTROL METHODS =========
    def update(self, agent: Agent, dt=1.0/5):
        """ This method is scheduled to be called repeatedly by the pyglet clock,
        where `dt` — the change in time since the last call.
        """
        dt = min(dt, 0.2) / agent.time_int_steps
        is_flying = agent.flying

        speed = FLYING_SPEED if is_flying else WALKING_SPEED
        speed_direction = _get_motion_direction(agent.strafe, agent.rotation, is_flying)
        self_motion_delta = _integrate_movement_over_time(speed, speed_direction, dt)

        agent.position, agent.dy, agent.time_int_steps = _update(
            agent.position, self_motion_delta, dt, agent.dy, is_flying,
            agent.time_int_steps, self.world, FACES
        )

        if not agent.sustain:
            agent.strafe = (0, 0)
            if agent.flying:
                agent.dy = 0

    def place_or_remove_block(self, agent, remove: bool, place: bool):
        if place == remove:
            return

        block, previous_block = self.hit_test(
            agent.position, get_sight_vector(agent.rotation)
        )

        if place and previous_block:
            self._try_place_block(agent, previous_block)
        if remove and block:
            self._try_remove_block(agent, block)

    def _try_place_block(self, agent, block_position: int_3d):
        texture = agent.active_block
        if agent.inventory[texture - 1] <= 0:
            return

        bx, by, bz = block_position
        if not _is_build_zone(bx, by, bz):
            return
        agent_position = to_float_3d(agent.position)
        if _is_agent_near(agent_position, bx, by, bz):
            return

        self.add_block(block_position, texture)
        agent.inventory[texture - 1] -= 1

    def _try_remove_block(self, agent, block_position: int_3d):
        texture = self.world[block_position]
        if texture == GREY or texture == WHITE:
            return
        self.remove_block(block_position)
        agent.inventory[texture - 1] += 1

    def get_focused_block(self, agent):
        vector = get_sight_vector(agent.rotation)
        return self.hit_test(agent.position, vector)[0]

    @staticmethod
    def move_camera(agent, dx: float, dy: float):
        agent.rotation = _move_camera(agent.rotation, dx, dy)

    @staticmethod
    def movement(agent, strafe: tuple[int, int], dy: float, inventory: Optional[int] = None):
        agent.strafe = _add_strafe(agent.strafe, strafe)
        agent.dy = _compute_dy(agent.dy, dy, agent.flying)

        if inventory is not None:
            if not (1 <= inventory <= 6):
                raise ValueError(f'Bad inventory id: {inventory}')
            agent.active_block = inventory
    # ========= END AGENT CONTROL =========

    # ========= UNIFIED AGENT CONTROL =========
    @staticmethod
    def parse_walking_discrete_action(action):
        # 0 noop; 1 forward; 2 back; 3 left; 4 right; 5 jump; 6-11 hotbar; 12 camera left;
        # 13 camera right; 14 camera up; 15 camera down; 16 attack; 17 use;
        # action = list(action).index(1)
        strafe = [0, 0]
        camera = [0, 0]
        dy = 0
        inventory = None
        remove = False
        add = False
        if action == 1:
            strafe[0] += -1
        elif action == 2:
            strafe[0] += 1
        elif action == 3:
            strafe[1] += -1
        elif action == 4:
            strafe[1] += 1
        elif action == 5:
            dy = 1
        elif 6 <= action <= 11:
            inventory = action - 5
        elif action == 12:
            camera[0] = -5
        elif action == 13:
            camera[0] = 5
        elif action == 14:
            camera[1] = -5
        elif action == 15:
            camera[1] = 5
        elif action == 16:
            remove = True
        elif action == 17:
            add = True
        return strafe, dy, inventory, camera, remove, add

    @staticmethod
    def parse_walking_action(action):
        strafe = [0,0]
        if action['forward']:
            strafe[0] += -1
        if action['back']:
            strafe[0] += 1
        if action['left']:
            strafe[1] += -1
        if action['right']:
            strafe[1] += 1
        jump = int(action['jump'])
        if action['hotbar'] == 0:
            inventory = None
        else:
            inventory = action['hotbar']
        camera = action['camera']
        remove = bool(action['attack'])
        add = bool(action['use'])
        return strafe, jump, inventory, camera, remove, add

    @staticmethod
    def parse_flying_action(action):
        """
        Args:
            action: dictionary with keys:
              * 'movement':  Box(low=-1, high=1, shape=(3,)) - forward/backward, left/right,
                  up/down movement
              * 'camera': Box(low=[-180, -90], high=[180, 90], shape=(2,)) - camera movement (yaw, pitch)
              * 'inventory': Discrete(7) - 0 for no-op, 1-6 for selecting block color
              * 'placement': Discrete(3) - 0 for no-op, 1 for placement, 2 for breaking
        """
        strafe = tuple(action['movement'][:2])
        dy = action['movement'][2]
        camera = list(action['camera'])
        inventory = action['inventory'] if action['inventory'] != 0 else None
        add = action['placement'] == 1
        remove = action['placement'] == 2
        return strafe, dy, inventory, camera, remove, add

    def step(self, agent, action, select_and_place=False, action_space='walking', discretize=True):
        if action_space == 'walking':
            if discretize:
                tup = self.parse_walking_discrete_action(action)
            else:
                tup = self.parse_walking_action(action)
        elif action_space == 'flying':
            tup = self.parse_flying_action(action)
        else:
            raise ValueError(f'Unknown action space: {action_space}')

        strafe, dy, inventory, camera, remove, add = tup
        if select_and_place and inventory is not None:
            add = True
            remove = False
        self.movement(agent, strafe=strafe, dy=dy, inventory=inventory)
        self.move_camera(agent, *camera)
        self.place_or_remove_block(agent, remove=remove, place=add)
        self.update(agent, dt=1/20.)

    # ========= END UNIFIED AGENT CONTROL =========


# ========= NUMBA-OPTIMIZED IMPLEMENTATIONS =========
@numba.jit(nopython=True, cache=True)
def _add_initial_blocks(blocks: dict[int_3d, int]):
    n = 18  # 1/2 width and height of world
    s = 1  # step size
    y = 0  # initial y height

    for x in range(-n, n + 1, s):
        for z in range(-n, n + 1, s):
            color = GREY if not _is_build_zone(x, y, z) else WHITE
            blocks[(x, y - 2, z)] = color


@numba.jit(nopython=True, cache=True, inline='always')
def _integrate_movement_over_time(v: float, v_direction: float_3d, dt: float) -> float_3d:
    # distance covered over time dt along the motion direction with speed v.
    dx, dy, dz = v_direction
    distance = v * dt
    return dx * distance, dy * distance, dz * distance


@numba.jit(nopython=True, cache=True)
def _handle_vertical_motion(agent_dy: float, dt: float) -> tuple[float, int]:
    # Update your vertical speed: if you are falling, speed up until you
    # hit terminal velocity; if you are jumping, slow down until you
    # start falling.
    agent_dy -= dt * GRAVITY
    time_int_steps = _n_update_steps_from_falling_speed(agent_dy)
    agent_dy = max(agent_dy, -TERMINAL_VELOCITY)
    return agent_dy, time_int_steps


@numba.jit(nopython=True, cache=True, inline='always')
def _n_update_steps_from_falling_speed(dy: float) -> int:
    if dy < -14:
        return 12
    if dy < -10:
        return 8
    if dy < -5:
        return 4
    return 2


@numba.jit(nopython=True, cache=True)
def get_sight_vector(rotation: float_2d) -> float_3d:
    """Returns the current line of sight vector indicating the direction the player is looking."""
    x, y = rotation
    # y ranges from -90 to 90, or -pi/2 to pi/2, so m ranges from 0 to 1 and
    # is 1 when looking ahead parallel to the ground and 0 when looking
    # straight up or down.
    m = math.cos(math.radians(y))
    # dy ranges from -1 to 1 and is -1 when looking straight down and 1 when
    # looking straight up.
    dy = math.sin(math.radians(y))
    dx = math.cos(math.radians(x - 90)) * m
    dz = math.sin(math.radians(x - 90)) * m
    return dx, dy, dz


@numba.jit(nopython=True, cache=True)
def _get_motion_direction(strafe: int_2d, rotation: float_2d, is_flying: bool) -> float_3d:
    """
    Returns the current motion vector indicating the velocity of the
    player: tuple containing the velocity in x, y, and z respectively.
    """
    # fb: forward/backward, lr: left/right
    strafe_fb, strafe_lr = strafe
    is_strafe = strafe_fb != 0 or strafe_lr != 0

    if not is_strafe:
        return 0., 0., 0.

    x, y = rotation
    strafe_degrees = math.degrees(math.atan2(strafe_fb, strafe_lr))
    y_angle = math.radians(y)
    x_angle = math.radians(x + strafe_degrees)
    dx = math.cos(x_angle)
    dz = math.sin(x_angle)
    if is_flying:
        m = math.cos(y_angle)
        dy = math.sin(y_angle)
        if strafe_lr:
            # Moving left or right.
            dy, m = 0.0, 1.0
        if strafe_fb > 0:
            # Moving backwards.
            dy *= -1
        # When you are flying up or down, you have less left and right motion.
        dx, dz = dx * m, dz * m
    else:
        dy = 0.0
    return dx, dy, dz


@numba.jit(nopython=True, cache=True, inline='always')
def _add_strafe(agent_strafe: int_2d, strafe: int_2d) -> int_2d:
    ag_strafe_fb, ag_strafe_lr = agent_strafe
    strafe_fb, strafe_lr = strafe
    return ag_strafe_fb + strafe_fb, ag_strafe_lr + strafe_lr


@numba.jit(nopython=True, cache=True, inline='always')
def _compute_dy(agent_dy: float, dy: float, agent_flying: bool) -> float:
    if dy != 0 and agent_dy == 0:
        return JUMP_SPEED * dy
    if agent_flying and dy == 0:
        return 0.


@numba.jit(nopython=True, cache=True, inline='always')
def _move_camera(rotation: float_2d, d_yaw: float, d_pitch: float) -> tuple[float, float]:
    yaw, pitch = rotation
    yaw, pitch = yaw + d_yaw, pitch + d_pitch

    pitch = max(-90., min(90., pitch))
    while yaw > 360.:
        yaw -= 360.
    while yaw < 0.0:
        yaw += 360.0

    return yaw, pitch


@numba.jit(nopython=True, cache=True, inline='always')
def _is_build_zone(x, y, z, pad=0):
    return -5 - pad <= x <= 5 + pad and -5 - pad <= z <= 5 + pad and -1 - pad <= y < 8 + pad


@numba.jit(nopython=True)
def _hit_test(
        world: dict[int_3d, int],
        position: float_3d, vector: float_3d,
        max_distance: int = 8
):
    """
    Line of sight search from current position. If a block is
    intersected it is returned, along with the block previously in the line
    of sight. If no block is found, return None, None.
    """
    if not world:
        return None, None

    x, y, z = position

    m = 5
    dx, dy, dz = vector
    dx, dy, dz = dx / m, dy / m, dz / m

    previous_block = None
    for i in range(max_distance * m):
        block = discretize_3d((x, y, z))
        should_check = i == 0 or block != previous_block

        if should_check and block in world:
            return block, previous_block

        previous_block = block
        x, y, z = x + dx, y + dy, z + dz

    return None, None


@numba.jit(nopython=True, cache=True)
def _is_agent_near(agent_position: float_3d, bx: int, by: int, bz: int) -> bool:
    x, y, z = agent_position
    y = y - (PLAYER_HEIGHT - 1.) + PLAYER_PAD
    bx -= 0.5
    bz -= 0.5
    return (
        bx <= x <= bx + 1
        and bz <= z <= bz + 1
        and (
            by <= y <= by + 1
            or by <= (y + 1) <= by + 1
        )
    )


@numba.jit(nopython=True, cache=True)
def _collide(
        position: float_3d, world: dict[int_3d, int], faces: list[int_3d]
) -> tuple[float_3d, bool]:
    stop_vertical = False

    new_position = list(position)
    block = discretize_3d(position)
    collide_candidate = list(block)

    # check all surrounding blocks
    for face_id, face in enumerate(faces):
        # up (0, 1, 0) and down (0, -1, 0) are face_id 0 and 1 respectively (it should be!)
        is_ground_or_ceiling = face_id <= 1

        # check each dimension independently
        for axis in range(3):
            bound_direction = face[axis]
            if not bound_direction:
                # unbound axis
                continue

            # how much overlap you have with this dimension.
            overlap = (new_position[axis] - block[axis]) * bound_direction
            if overlap < PLAYER_PAD:
                # you are not touching the block in this dimension.
                continue

            collide_candidate[axis] += bound_direction
            x, y, z = collide_candidate
            is_collided = _check_collision(x, y, z, world)
            collide_candidate[axis] -= bound_direction

            stop_vertical |= is_collided and is_ground_or_ceiling
            if is_collided:
                new_position[axis] -= bound_direction * (overlap - PLAYER_PAD)

    x, y, z = new_position
    return (x, y, z), stop_vertical


@numba.jit(nopython=True, cache=True)
def _check_collision(x: int, y: int, z: int, world: dict[int_3d, int]) -> bool:
    # traverse sequentially over the height of the player
    for _ in range(PLAYER_HEIGHT):
        if (x, y, z) in world:
            return True
        y -= 1
    return False


@numba.jit(nopython=True, cache=True)
def _update(
        agent_position: float_3d, self_motion_delta: float_3d, dt: float,
        agent_dy: float, is_flying: bool, agent_time_int_steps: int,
        world, faces
):
    for _ in range(agent_time_int_steps):
        if not is_flying:
            # apply gravity to agent
            agent_dy, agent_time_int_steps = _handle_vertical_motion(agent_dy, dt)

        # distance covered this tick via self-motion
        dx, dy, dz = self_motion_delta
        # add the gravity-induced motion
        dy += agent_dy * dt

        # calc new position + check collisions
        stop_vertical = False
        x, y, z = agent_position
        _x, _y, _z = x + dx, y + dy, z + dz

        if _is_build_zone(_x, _y, _z, pad=2):
            (x, y, z), stop_vertical = _collide((_x, _y, _z), world, faces)
        elif not is_flying:
            (x, y, z), stop_vertical = _collide((x, y + dy, z), world, faces)

        agent_position = (x, y, z)
        if stop_vertical:
            agent_dy = 0

    return agent_position, agent_dy, agent_time_int_steps
