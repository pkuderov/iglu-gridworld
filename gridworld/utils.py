import math
import numba

TICKS_PER_SEC = 60000

# Size of sectors used to ease block loading.
SECTOR_SIZE = 16

WALKING_SPEED = 5
FLYING_SPEED = 15

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

TERMINAL_VELOCITY = 50

PLAYER_HEIGHT = 2


@numba.jit(nopython=True, cache=True, inline='always')
def cube_vertices(x, y, z, n, top_only=False):
    """ Return the vertices of the cube at position x, y, z with size 2*n.

    """
    if top_only:
        return [
            x - n, y + n, z - n, x - n, y + n, z + n, x + n, y + n, z + n, x + n, y + n, z - n,
            # top
        ]
    return [
        x - n, y + n, z - n, x - n, y + n, z + n, x + n, y + n, z + n, x + n, y + n, z - n,  # top
        x - n, y - n, z - n, x + n, y - n, z - n, x + n, y - n, z + n, x - n, y - n, z + n,
        # bottom
        x - n, y - n, z - n, x - n, y - n, z + n, x - n, y + n, z + n, x - n, y + n, z - n,  # left
        x + n, y - n, z + n, x + n, y - n, z - n, x + n, y + n, z - n, x + n, y + n, z + n,  # right
        x - n, y - n, z + n, x + n, y - n, z + n, x + n, y + n, z + n, x - n, y + n, z + n,  # front
        x + n, y - n, z - n, x - n, y - n, z - n, x - n, y + n, z - n, x + n, y + n, z - n,  # back
    ]


@numba.jit(nopython=True, cache=True)
def cube_normals(top_only=False):
    if top_only:
        return [
            0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,  # top
        ]
    return [
        0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,  # top
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,  # bottom
        -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,  # left
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,  # right
        0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,  # front
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,  # back
    ]


@numba.jit
def normalize(position: tuple[float, float, float]) -> tuple[int, int, int]:
    """ Accepts `position` of arbitrary precision and returns the block
    containing that position.

    Parameters
    ----------
    position : tuple of len 3

    Returns
    -------
    block_position : tuple of ints of len 3

    """
    x, y, z = position
    x, y, z = round(x), round(y), round(z)
    return x, y, z


@numba.jit(nopython=True, cache=True)
def tex_coord(x, y, split=False, side_n=0) -> list[float]:
    """ Return the bounding vertices of the texture square.

    """
    n = 4
    m = 1.0 / n
    m1 = 1.0 / n / (2 if split else 1)

    cx = 0.
    cy = 0.

    if split:
        if side_n == 0:
            cx, cy = 0, 0
        elif side_n == 1:
            cx, cy = 0, 0.125
        elif side_n == 2:
            cx, cy = 0.125, 0
        elif side_n == 3:
            cx, cy = 0.125, 0.125
    else:
        cx, cy = 0, 0
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
    """ Return a list of the texture squares for the top, bottom and side.

    """
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
def _tex_coords(x: int, y: int):
    return texture_coordinates(x, y, False, False)


@numba.jit(nopython=True)
def _tex_coords_split(x: int, y: int):
    return texture_coordinates(x, y, False, True)


@numba.jit(nopython=True)
def _tex_coords_top(x: int, y: int):
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
    WHITE: _tex_coords(0, 0),
    GREY: _tex_coords(1, 0),
    BLUE: _tex_coords_split(2, 0),
    GREEN: _tex_coords_split(3, 0),
    RED: _tex_coords_split(0, 1),
    ORANGE: _tex_coords_split(1, 1),
    PURPLE: _tex_coords_split(2, 1),
    YELLOW: _tex_coords_split(3, 1)
}

id2top_texture = {
    WHITE: _tex_coords_top(0, 0),
    GREY: _tex_coords_top(1, 0),
    BLUE: _tex_coords_top(2, 0),
    GREEN: _tex_coords_top(3, 0),
    RED: _tex_coords_top(0, 1),
    ORANGE: _tex_coords_top(1, 1),
    PURPLE: _tex_coords_top(2, 1),
    YELLOW: _tex_coords_top(3, 1)
}

FACES = [
    (0, 1, 0),
    (0, -1, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (0, 0, -1),
]

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Z = 11
BUILD_ZONE_SIZE = 9, 11, 11
