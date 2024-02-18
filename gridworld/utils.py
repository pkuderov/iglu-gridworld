from __future__ import annotations

import math
import numba
from numba.typed.typedlist import List

TICKS_PER_SEC = 60000

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11

PLAYER_HEIGHT = 2

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16

WALKING_SPEED = 5
FLYING_SPEED = 15
TERMINAL_VELOCITY = 50

GRAVITY = 20.0
MAX_JUMP_HEIGHT = 1.2  # About the height of a block.

# To derive the formula for calculating jump speed, first solve
#    v_t = v_0 + a * t
# for the time at which you achieve maximum height, where a is the acceleration
# due to gravity and v_t = 0. This gives:
#    t = - v_0 / a
# Use t and the desired MAX_JUMP_HEIGHT to solve for v_0 (jump speed) in
#    s = s_0 + v_0 * t + (a * t^2) / 2
JUMP_SPEED = math.sqrt(2 * GRAVITY * MAX_JUMP_HEIGHT)

int_2d = tuple[int, int]
int_3d = tuple[int, int, int]
float_2d = tuple[float, float]
float_3d = tuple[float, float, float]


@numba.jit(nopython=True, cache=True, inline='always')
def cube_vertices(x, y, z, n, top_only=False):
    """ Return the vertices of the cube at position x, y, z with size 2*n."""
    if top_only:
        return [
            # top
            x - n, y + n, z - n,
            x - n, y + n, z + n,
            x + n, y + n, z + n,
            x + n, y + n, z - n,
        ]
    return [
        # top
        x - n, y + n, z - n,
        x - n, y + n, z + n,
        x + n, y + n, z + n,
        x + n, y + n, z - n,
        # bottom
        x - n, y - n, z - n,
        x + n, y - n, z - n,
        x + n, y - n, z + n,
        x - n, y - n, z + n,
        # left
        x - n, y - n, z - n,
        x - n, y - n, z + n,
        x - n, y + n, z + n,
        x - n, y + n, z - n,
        # right
        x + n, y - n, z + n,
        x + n, y - n, z - n,
        x + n, y + n, z - n,
        x + n, y + n, z + n,
        # front
        x - n, y - n, z + n,
        x + n, y - n, z + n,
        x + n, y + n, z + n,
        x - n, y + n, z + n,
        # back
        x + n, y - n, z - n,
        x - n, y - n, z - n,
        x - n, y + n, z - n,
        x + n, y + n, z - n,
    ]


@numba.jit(nopython=True, cache=True, inline='always')
def discretize_3d(position: float_3d) -> int_3d:
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position."""
    x, y, z = position
    x, y, z = round(x), round(y), round(z)
    return x, y, z


def to_int_3d(position: tuple | list) -> int_3d:
    # use this to make tuple of homogeneous int type in case it's np.int64 for example
    x, y, z = position
    return int(x), int(y), int(z)


def to_float_3d(position: tuple | list) -> float_3d:
    # use this to make tuple of homogeneous float type in case it's np.float64 for example
    x, y, z = position
    return float(x), float(y), float(z)


@numba.jit(nopython=True, cache=True)
def tex_coord(x, y, split=False, side_n=0) -> list[float]:
    """ Return the bounding vertices of the texture square."""
    n = 4
    m = 1.0 / n
    m1 = m / (2 if split else 1)

    cx, cy = 0., 0.

    if split:
        if side_n == 0:
            cx, cy = 0, 0
        elif side_n == 1:
            cx, cy = 0, 0.125
        elif side_n == 2:
            cx, cy = 0.125, 0
        elif side_n == 3:
            cx, cy = 0.125, 0.125

    dx = x * m
    dy = y * m
    return [
        cx + dx, cy + dy,
        cx + dx + m1, cy + dy,
        cx + dx + m1, cy + dy + m1,
        cx + dx, cy + dy + m1
    ]


@numba.jit(nopython=True)
def texture_coordinates(x: int, y: int, top_only: bool, split: bool) -> list[float]:
    """ Return a list of the texture squares for the top, bottom and side."""
    result: list[float] = []
    if split:
        if top_only:
            return [float(x), float(y)]
        else:
            result += tex_coord(x, y, split, side_n=1)
            result += tex_coord(x, y, split, side_n=2)
            result += tex_coord(x, y, split, side_n=0)
            result += tex_coord(x, y, split, side_n=0)
            result += tex_coord(x, y, split, side_n=3)
            result += tex_coord(x, y, split, side_n=3)
    else:
        side = tex_coord(x, y, False, 0)
        for _ in range(1 if top_only else 6):
            result.extend(side)
    return result


@numba.jit(nopython=True)
def _texture_coordinates_default(x: int, y: int):
    return texture_coordinates(x, y, False, False)


@numba.jit(nopython=True)
def _texture_coordinates_split(x: int, y: int):
    return texture_coordinates(x, y, False, True)


@numba.jit(nopython=True)
def _texture_coordinates_top(x: int, y: int):
    return texture_coordinates(x, y, True, False)


WHITE = -1
GREY = 0
BLUE = 1
GREEN = 2
RED = 3
ORANGE = 4
PURPLE = 5
YELLOW = 6

id2texture = {
    WHITE: _texture_coordinates_default(0, 0),
    GREY: _texture_coordinates_default(1, 0),
    BLUE: _texture_coordinates_split(2, 0),
    GREEN: _texture_coordinates_split(3, 0),
    RED: _texture_coordinates_split(0, 1),
    ORANGE: _texture_coordinates_split(1, 1),
    PURPLE: _texture_coordinates_split(2, 1),
    YELLOW: _texture_coordinates_split(3, 1)
}

id2top_texture = {
    WHITE: _texture_coordinates_top(0, 0),
    GREY: _texture_coordinates_top(1, 0),
    BLUE: _texture_coordinates_top(2, 0),
    GREEN: _texture_coordinates_top(3, 0),
    RED: _texture_coordinates_top(0, 1),
    ORANGE: _texture_coordinates_top(1, 1),
    PURPLE: _texture_coordinates_top(2, 1),
    YELLOW: _texture_coordinates_top(3, 1)
}

# The six cardinal directions: up, down, north, south, east, west.
# NB: The order of the directions is important.
FACES = List([
    (0, 1, 0),
    (0, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
])
