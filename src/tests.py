import numpy as np
from scipy.spatial.transform import Rotation


pos = np.array([3, 3, 3])
size = np.array([3, 3, 3])
points = np.empty((8, 3), float)
points[:, 0] = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * size[0] / 2
points[:, 1] = np.array([-1, -1, 1, 1, 1, 1, -1, -1]) * size[1] / 2
points[:, 2] = np.array([-1, 1, -1, 1, -1, 1, -1, 1]) * size[2] / 2
print(points)
color = "white"
print(np.full(12, color))


class C:
    def f(self, x):
        x[0] = 2
        print(x)


a = np.array([1, 2, 3, 4])
c = C()
c.f(a)
print(a)


def perspective(elements):
    # for i in range(np.size(elements, axis=0)):
    #     for j in range(np.size(elements, axis=1)):
    condition = elements[:, :, 1] > 0
    elements[:, :, 1] = np.where(condition, elements[:, :, 1], 0.001)
    return elements[:, :, [0, 2]] / elements[:, :, 1].reshape(np.size(elements, axis=0), -1, 1)


a = np.array([[[1.0, 0.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
              [[1.0, 0.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
# print(np.size(a))
# print(perspective(a))


def rotate(elements):
    rotation_x = Rotation.from_euler('x', 0).as_matrix()
    rotation_y = Rotation.from_euler('y', 0).as_matrix()
    rotation_z = Rotation.from_euler('z', 1.57).as_matrix()
    rotation_total = rotation_x @ rotation_y @ rotation_z
    # print(rotation_x @ rotation_y @ rotation_z)
    # print(rotation_z @ rotation_y @ rotation_x)
    return np.transpose(rotation_total @ np.transpose(elements, axes=(0, 2, 1)), axes=(0, 2, 1))


# print(rotate(a))

def sort(elements, colors):
    polygon_centers = np.mean(elements, axis=1)
    distances_to_camera = np.linalg.norm(polygon_centers - camera_position, axis=1)
    sorted_indices = np.argsort(distances_to_camera)
    sorted_elements = elements[sorted_indices]
    sorted_colors = colors[sorted_indices]
    return sorted_elements, sorted_colors


camera_position = np.array([0, 5, 0])
elements = np.array([[[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]],
                     [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]],
                     [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]])
colors = np.array(["red", "blue", "green"])

sorted_elements, sorted_colors = sort(elements, colors)

print("Sorted elements:")
print(sorted_elements)
print("Sorted colors:")
print(sorted_colors)

elements_axis1_size = 8
print(np.array([(i + elements_axis1_size * (i % 2)) // 2
                for i in range(elements_axis1_size)]) == np.array([0, 4, 1, 5, 2, 6, 3, 7]))
