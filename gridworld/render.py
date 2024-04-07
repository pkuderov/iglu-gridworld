import math
import os
import platform
import time

import numba
import numpy as np
import pyglet
from filelock import FileLock

pyglet.options['shadow_window'] = False
if os.environ.get('IGLU_HEADLESS', '1') == '1':
    import pyglet
    pyglet.options["headless"] = True
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if devices is not None and devices != '':
        pyglet.options['headless_device'] = int(devices.split(',')[-1])


from pyglet import app, image
from pyglet.gl import (
    glClearColor, glEnable, GL_CULL_FACE, glTexParameteri, GL_TEXTURE_2D,
    GL_TEXTURE_MIN_FILTER, GL_NEAREST, GL_TEXTURE_MAG_FILTER, glViewport, glMatrixMode,
    glLoadIdentity, gluPerspective, glRotatef, glTranslatef, GL_MODELVIEW, GL_PROJECTION,
    glColor3d, GL_QUADS, GL_DEPTH_TEST
)
from pyglet.graphics import Batch, TextureGroup
from pyglet.window import Window

from gridworld.utils import WHITE, GREY, cube_vertices, id2texture, id2top_texture

_60FPS = 1./60
PLATFORM = platform.system()


class Renderer(Window):
    TEXTURE_PATH = 'texture.png'

    def __init__(self, width, height, dir_path, invert_y=False, **kwargs):
        # workaround for non-headless macOS
        is_headless = pyglet.options['headless']
        raw_width, raw_height = width, height
        width, height = _get_img_size(width, height, is_headless)

        super().__init__(width=width, height=height, **kwargs)

        texture_path = os.path.join(dir_path, Renderer.TEXTURE_PATH)
        with FileLock(f'/tmp/iglu_lock'):
            self.texture_group = TextureGroup(image.load(texture_path).get_texture())

        self.batch = Batch()
        self._shown = {}

        self.cube_vertices_cache = {}

        self.buffer_manager = pyglet.image.get_buffer_manager()
        self.invert_y = invert_y
        self.last_frame_dt = 0
        self.realtime_rendering = os.environ.get('IGLU_RENDER_REALTIME', '0') == '1'

        self.is_headless = is_headless
        if not self.is_headless:
            app.platform_event_loop.start()
            self.dispatch_event('on_enter')

        # precompute rendered shapes. I assume that the window size will not change
        self.render_shape = (raw_height, raw_width, 4)

        setup()

    def set_3d(self, agent_position, agent_rotation):
        """ Configure OpenGL to draw in 3d."""
        width, height = self.get_size()
        glEnable(GL_DEPTH_TEST)
        viewport = self.get_viewport_size()
        _set_3d(viewport, width, height, agent_position, agent_rotation)

    # noinspection PyMethodOverriding
    def on_draw(self, agent_position, agent_rotation):
        """ Called by pyglet to draw the canvas."""
        self.clear()
        self.set_3d(agent_position, agent_rotation)
        glColor3d(1, 1, 1)
        self.batch.draw()

    def render(self, agent_position, agent_rotation):
        if not self.is_headless:
            t = time.perf_counter()
            self.switch_to()

        self.on_draw(agent_position, agent_rotation)

        data = (
            self.buffer_manager
            .get_color_buffer()
            .get_image_data()
            .get_data()
        )
        # no copies are made here
        rendered = np.asarray(data, dtype=np.uint8).reshape(self.render_shape)
        # NB: drop alpha channel
        rendered = rendered[..., :3]
        if self.invert_y:
            # WARN: for faster rendering, turn off flipping the Y-axis (0-th)
            rendered = rendered[::-1]

        if not self.is_headless:
            # noinspection PyUnboundLocalVariable
            dt = time.perf_counter() - t
            self.last_frame_dt += dt
            if self.realtime_rendering:
                self.flip()
            if self.last_frame_dt >= _60FPS:
                if not self.realtime_rendering:
                    self.flip()
                app.platform_event_loop.step(dt)
                self.last_frame_dt = 0.

        return rendered

    def add_block(self, position, texture_id, **_):
        x, y, z = position
        top_only = texture_id == WHITE or texture_id == GREY
        texture = id2top_texture[texture_id] if top_only else id2texture[texture_id]

        cube_vertex_key = (x, y, z, top_only)
        vertex_data = self.cube_vertices_cache.get(cube_vertex_key, None)
        if vertex_data is None:
            vertex_data = self.cube_vertices_cache[cube_vertex_key] = cube_vertices(
                x, y, z, 0.5, top_only=top_only
            )

        # create vertex list
        # FIXME Maybe `add_indexed()` should be used instead
        self._shown[position] = self.batch.add(
            4 if top_only else 24,
            GL_QUADS, self.texture_group,
            ('v3f/static', vertex_data),
            ('t2f/static', texture),
        )

    def remove_block(self, position, **_):
        if position in self._shown:
            self._shown.pop(position).delete()

    def close(self):
        if not self.is_headless:
            app.platform_event_loop.stop()
            for key in self._shown:
                self._shown.pop(key).delete()

        super().close()


@numba.jit(nopython=True)
def setup():
    """ Basic OpenGL configuration."""
    glClearColor(0.5, 0.69, 1.0, 1)
    glEnable(GL_CULL_FACE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)


@numba.jit(nopython=True)
def _set_3d(viewport, width, height, agent_position, agent_rotation):
    glViewport(0, 0, max(1, viewport[0]), max(1, viewport[1]))
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(90.0, width / float(height), 0.1, 30.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    x, y = agent_rotation
    glRotatef(x, 0, 1, 0)
    x_radians = math.radians(x)
    glRotatef(-y, math.cos(x_radians), 0, math.sin(x_radians))
    x, y, z = agent_position
    glTranslatef(-x, -y, -z)


def _get_img_size(width, height, is_headless: bool):
    if PLATFORM == 'Darwin' and not is_headless:
        # for some reason COCOA renderer sets the viewport size
        # to be twice as the requested size. This is a temporary
        # workaround
        scale = 2
    else:
        scale = 1

    return width // scale, height // scale
