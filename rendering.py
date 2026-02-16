import math
import numpy as np
from scipy.spatial.transform import Rotation
import pygame
from screeninfo import get_monitors
import gameObjects


# Render the playfield on the display
class Renderer:
    def __init__(self, _camera, _objects, width=800, height=600, background=(20, 20, 20)):
        self._camera = _camera  # knows the camera to render from outlooks
        self._objects = _objects  # reference to the game's objects
        pygame.init()
        # window settings
        self.window_width = width
        self.window_height = height
        self.background = background
        self.full_screen = False
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("3D Perspective Projection")

    def switch_screen(self):
        if self.full_screen:
            self.window = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
            self.full_screen = False
        else:
            monitor = get_monitors()[0]
            self.window = pygame.display.set_mode((monitor.width, monitor.height), pygame.FULLSCREEN)
            self.full_screen = True

    # generate 3D points, rotate vertices, optimize the order of rendering, convert 3D vertices to 2D and draw them
    def update(self):
        self.window.fill(self.background)  # render background
        elements, colors = self.select(self._objects)
        self.rotate(elements)
        self.sort(elements, colors)
        self.perspective(elements)
        self.draw(elements, colors)
        pygame.display.update()  # render display

    # select all elements of the objects
    def select(self, objects):
        polygons_4 = np.empty((0,), float)
        colors_4 = np.empty((0,), tuple)
        for obj in objects:
            if not isinstance(obj.pos, np.ndarray):
                raise "position of block is not an array"
            if np.shape(obj.pos)[0] != 3:
                raise "position of block is not 3-dimensional"
            if isinstance(obj, gameObjects.Cube):
                new_faces, new_colors = self.generateCube(obj.pos, obj.size, obj.color)
                polygons_4 = np.append(polygons_4, new_faces)
                colors_4 = np.append(colors_4, new_colors)
            elif isinstance(obj, gameObjects.Floor):
                new_face, new_color = self.generateFloor(obj.pos, obj.size, obj.color)
                polygons_4 = np.append(polygons_4, new_face)
                colors_4 = np.append(colors_4, new_color)
        polygons_4 = polygons_4.reshape(-1, 4, 3) - self._camera.center
        colors_4 = colors_4.reshape(-1, 3)
        polygons = [polygons_4]
        colors = [colors_4]
        for form in polygons:
            form[:, :, 2] *= -1
        return polygons, colors

    # rotate the position of vertices to the camera
    def rotate(self, elements):
        # radian measure
        rotation_x = Rotation.from_euler('x', self._camera.rotation[0]).as_matrix()
        rotation_y = Rotation.from_euler('y', self._camera.rotation[1]).as_matrix()
        rotation_z = Rotation.from_euler('z', self._camera.rotation[2]).as_matrix()
        rotation_total = rotation_x @ rotation_y @ rotation_z
        for i in range(len(elements)):
            elements[i] = np.transpose(rotation_total @ np.transpose(elements[i], axes=(0, 2, 1)), axes=(0, 2, 1))

    # sort the elements by its distance to the camera
    def sort(self, elements, colors):
        for i in range(len(elements)):
            # compute the coordinates seen by the camera
            elements[i] += np.array([0, self._camera.scope, 0])
            # delete the polygons that are behind the camera
            invalid_polygons = np.empty((0,), int)
            for p in range(np.size(elements[i], 0)):
                for v in range(np.size(elements[i], 1)):
                    if elements[i][p, v, 1] <= 0:
                        invalid_polygons = np.append(invalid_polygons, p)
            elements[i] = np.delete(elements[i], invalid_polygons, 0)
            colors[i] = np.delete(colors[i], invalid_polygons, 0)
            # sort the valid polygons
            polygon_centers = np.mean(elements[i], axis=1)
            camera_distances = np.linalg.norm(polygon_centers, axis=1)
            sorted_indices = np.argsort(camera_distances)[::-1]
            elements[i] = elements[i][sorted_indices]
            colors[i] = colors[i][sorted_indices]

    # convert 3D vertices to 2D
    def perspective(self, elements):
        """ perspective projection """
        for i in range(len(elements)):
            # calculate the projected x- and z-coordinates using inverse perspective projection
            form = elements[i].shape[1]
            elements[i] = np.column_stack((elements[i][:, :, 0] / elements[i][:, :, 1],
                                           elements[i][:, :, 2] / elements[i][:, :, 1]))
            polygons_size = np.size(elements[i], 1)
            elements[i] = elements[i][:, np.array([(i + polygons_size * (i % 2)) // 2 for i in range(polygons_size)])]
            elements[i] = np.reshape(elements[i], (-1, form, 2))
            # scale the projected coordinates based on the image depth
            elements[i] *= self._camera.depth

            ''' simple and not perspective projection:
            condition = elements[i][:, :, 1] + self.camera.scope > 0
            elements[i][:, :, 1] = np.where(condition, elements[i][:, :, 1], 0.001)
            elements[i] = elements[i][:, :, [0, 2]] / (
                    elements[i][:, :, 1].reshape(np.size(elements[i], axis=0), -1, 1) + self.camera.scope) '''
            ''' Distorted field of view with inward curvature:
            # Calculate the tangent of the half field of view
            tan_half_fov = np.tan(self.camera.fov / 2)
            # Calculate the distance of the point to the camera
            distance = np.linalg.norm(elements[i][:, :])
            # Calculate the projected x- and z-coordinates based on the distance and the field of view
            projected_x = elements[i][:, :, 0] / (distance * tan_half_fov)
            projected_z = elements[i][:, :, 2] / (distance * tan_half_fov) '''

    # draw vertices, edges and faces on a 2D display
    def draw(self, elements, colors):  # TODO: recognize different forms of polygons
        screen_info = pygame.display.Info()
        # define pointer for elements and its final state
        pointer = [0 for _ in range(len(elements))]
        final_state = [np.size(e, 0) for e in elements]
        while pointer != final_state:
            # select the next polygon and draw its edges and vertices
            polygon = elements[0][pointer[0]].view()
            for i in range(np.size(polygon, 0)):
                pygame.draw.line(self.window, colors[0][pointer[0]],
                                 polygon[i] + np.array([screen_info.current_w / 2, screen_info.current_h / 2]),
                                 polygon[(i + 1) % np.size(polygon, 0)]
                                 + np.array([screen_info.current_w / 2, screen_info.current_h / 2]), 2)
                pygame.draw.circle(self.window, "black",
                                   polygon[i] + np.array([screen_info.current_w / 2, screen_info.current_h / 2]), 2)
            pointer[0] += 1

    # Following methods are used to render game objects:

    # generate the vertices, edges and areas of a block
    @staticmethod
    def generateCube(pos, size, color):
        points = np.array([pos + (np.array([x, y, z]) * size / 2) for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
        '''edges = (np.array([points[0], points[1], points[0], points[2], points[0], points[4], points[1], points[3],
                           points[1], points[5], points[2], points[3], points[2], points[6], points[3], points[7],
                           points[4], points[5], points[4], points[6], points[5], points[7], points[6], points[7]])'''
        faces = np.array(
            [points[0], points[1], points[3], points[2], points[4], points[5], points[7], points[6], points[0],
             points[1], points[5], points[4], points[2], points[3], points[7], points[6], points[0], points[2],
             points[6], points[4], points[1], points[3], points[7], points[5]])
        colors = np.full((6, 3), color)
        return faces, colors

    @staticmethod
    def generateFloor(pos, size, color):
        face = np.array([pos + (np.array([x, y, 0]) * size / 2) for x in [-1, 1] for y in [-1, 1]])[np.array([0, 1, 3, 2])]
        color = np.array([color])
        return face, color

    # generate the vertices of a globe with precision TODO: create a globe
    @staticmethod
    def generateGlobe(num):
        def refine(points, mesh):
            n = points.shape[0]
            m = mesh.shape[0]
            nn = n + 3 * m // 2
            mm = 4 * m
            newpoints = np.zeros((nn, 3))
            newmesh = np.zeros((mm, 3), dtype=int)
            newpoints[:n] = points
            splits = {}
            j = 0
            for tri in mesh:
                i = [0, 0, 0]
                for k in range(3):
                    l = (k + 1) % 3
                    key = (tri[k], tri[l]) if k < l else (tri[l], tri[k])
                    if key in splits:
                        i[k] = splits[key]
                    else:
                        i[k] = n + len(splits)
                        splits[key] = i[k]
                        x = points[tri[k]] + points[tri[l]]
                        x /= np.linalg.norm(x)
                        newpoints[i[k]] = x
                newmesh[j] = [tri[0], i[0], i[2]]
                j += 1
                newmesh[j] = [tri[1], i[0], i[1]]
                j += 1
                newmesh[j] = [tri[2], i[1], i[2]]
                j += 1
                newmesh[j] = i
                j += 1
            return newpoints, newmesh

        gsr = (1 + np.sqrt(5)) / 2
        points = np.array([
            [+gsr, +1, 0], [+gsr, -1, 0], [-gsr, -1, 0], [-gsr, +1, 0],
            [0, +gsr, +1], [0, +gsr, -1], [0, -gsr, -1], [0, -gsr, +1],
            [+1, 0, +gsr], [-1, 0, +gsr], [-1, 0, -gsr], [+1, 0, -gsr]
        ]) / np.sqrt(1 + gsr ** 2)
        mesh = np.array([
            [0, 1, 8], [0, 1, 11], [2, 3, 9], [2, 3, 10], [0, 4, 5],
            [3, 4, 5], [1, 6, 7], [2, 6, 7], [4, 8, 9], [7, 8, 9],
            [5, 10, 11], [6, 10, 11], [0, 4, 8], [0, 5, 11], [1, 7, 8],
            [1, 6, 11], [3, 4, 9], [3, 5, 10], [2, 7, 9], [2, 6, 10],
        ], dtype=int)
        for _ in range(num):
            points, mesh = refine(points, mesh)
        order = np.argsort(points @ [1e-5, 3e-5, -1])
        return points[order]

    # Following methods are used to implement a shader: TODO

    # calculate the brightness effected by a light (l, h, k), reflected by the norm vector (n)
    # and seen from the camera (c) for a point (x)
    def brightness(self, x, n, l, h, k):  # point, norm vector, light, brightness, refraction
        if not (isinstance(x, np.ndarray) and isinstance(n, np.ndarray) and isinstance(l, np.ndarray)):
            raise "x, n or l is not an array"
        if np.size(x) != 3 or np.size(n) != 3 or np.size(l) != 3:
            raise "x, n or l is not a 3-dimensional point or vector"
        c = np.array([0, self._camera.scope, 0])
        v = (x - c) / np.linalg.norm(x - c)  # (point - point) / norm vector
        s = l - x  # point - point
        t = s - (2 * (n @ s)) * n  # vector - num * dot product * vector
        m = t / np.linalg.norm(t)  # vector / norm vector
        return h * (v @ m) ** k / np.linalg.norm(x - l)  # num * dot product ** number / norm vector

    # calculate the color of vertices influenced by all lights
    def color(self, points):
        if not isinstance(points, np.ndarray):
            raise "points are not arrays"
        light = np.array([])
        for x in points:
            light = np.append(light, [self.brightness(x, x, np.array([0, 0, -3]), np.array([-50, 0, 0]), 50, 1)
                                      + self.brightness(x, x, np.array([0, 0, -3]), np.array([2, 5, 1]), 25, 4)
                                      + self.brightness(x, x, np.array([0, 0, -3]), np.array([-4, -5, 0]), 6, 2)])
        return light.reshape(-1, 1)


class Camera:
    def __init__(self, center=(0, 0, 0), rotation=(0, 0, 0), scope=50, depth=500, outlook="third_person"):
        self.movements = {"right": np.array([1, 1, 0]), "left": np.array([-1, -1, 0]),
                          "forward": np.array([1, 1, 0]), "backward": np.array([-1, -1, 0]),
                          "up": np.array([0, 0, 1]), "down": np.array([0, 0, -1])}
        self.turns = {"right": np.array([0, 0, 1]), "left": np.array([0, 0, -1]),
                      "up": np.array([1, 0, 0]), "down": np.array([-1, 0, 0])}
        self.zooms = {"+": -1, "-": 1}
        self.outlooks = {"first_person": 0, "third_person": 0}
        self.center = np.empty((3,))
        self.rotation = np.empty((3,))
        self.scope = 0
        self.outlook = ""
        self.depth = 0
        # self.fov = fov
        self.set_center(center)
        self.set_rotation(rotation)
        self.set_scope(scope)
        self.set_outlook(outlook)
        self.set_depth(depth)
        self.reset_info = [self.center, self.rotation, self.scope, self.outlook, self.depth]

    def move(self, direction, speed):
        if not (direction in self.movements and isinstance(speed, (int, float))):
            raise "camera tries an invalid movement"
        rotation = 1
        if direction in ["right", "left"]:
            rotation = np.array([math.cos(-self.rotation[2]), math.sin(-self.rotation[2]), 0])
        elif direction in ["forward", "backward"]:
            rotation = np.array([math.sin(self.rotation[2]), math.cos(self.rotation[2]), 0])
        self.set_center(self.center + self.movements[direction] * rotation * speed)

    def turn(self, direction, speed):
        if not (direction in self.turns and isinstance(speed, (int, float))):
            raise "camera tries an invalid turn"
        self.set_rotation(self.rotation + self.turns[direction] * speed * np.pi / 360)

    def zoom(self, direction, speed):
        if not (direction in self.zooms and isinstance(speed, (int, float))):
            raise "Camera tries an invalid zoom"
        self.set_scope(self.scope * (1 + self.zooms[direction] * speed))

    def change_outlook(self):
        outlooks = list(self.outlooks.keys())
        self.set_outlook(outlooks[(outlooks.index(self.outlook) + 1) % len(self.outlooks)])

    def set_all(self, center=(0, 0, 0), rotation=(0, 0, 0), scope=50, depth=500, outlook="third_person"):
        self.set_center(center)
        self.set_rotation(rotation)
        self.set_scope(scope)
        self.set_outlook(depth)
        self.set_depth(outlook)

    def reset(self):
        self.set_center(self.reset_info[0])
        self.set_rotation(self.reset_info[1])
        self.set_scope(self.reset_info[2])
        self.set_outlook(self.reset_info[3])
        self.set_depth(self.reset_info[4])

    def set_center(self, center):
        if not isinstance(center, (tuple, list, np.ndarray)):
            raise "camera tries to set an invalid center"
        self.center = np.array(center)

    def set_rotation(self, rotation):  # y-rotation is always 0 caused by rotation view
        if not isinstance(rotation, (tuple, list, np.ndarray)):
            raise "camera tries to set an invalid center"
        self.rotation = np.array([np.clip(rotation[0], -np.pi / 2, np.pi / 2), 0, rotation[2] % (2 * np.pi)])

    def set_scope(self, scope):
        if not isinstance(scope, (int, float)):
            raise "camera tries to set an invalid scope"
        self.scope = max(1, scope)

    def set_outlook(self, outlook):
        if outlook not in self.outlooks:
            raise "camera tries an invalid outlook"
        if self.outlook == "":
            self.outlooks["third_person"] = self.scope  # initialize the third person's scope
        else:
            self.outlooks[self.outlook] = self.scope  # save the last outlook's scope
        self.set_scope(self.outlooks[outlook])  # set the new outlook's scope
        self.outlook = outlook

    def set_depth(self, depth):
        if not isinstance(depth, (int, float)):
            raise "camera tries to set an invalid scope"
        self.depth = depth

