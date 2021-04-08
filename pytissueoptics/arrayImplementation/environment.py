from typing import NamedTuple
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

class Components(NamedTuple):
    sources: list
    geometries: list


class Environment:
    def __init__(self):
        self.voxels = None
        self.boundaries = None
        self.source = None
        self.components = None

    def place(self, component, position):
        self.components.append

    def setBoundaries(self, x, y, z):
        self.boundaries = [x, y, z]

    def setVoxelsResolution(self, x, y, z):
        x = np.linspace(self.boundaries[0][0], self.boundaries[0][1], x)
        y = np.linspace(self.boundaries[1][0], self.boundaries[1][1], y)
        z = np.linspace(self.boundaries[2][0], self.boundaries[2][1], z)
        xv, yv, zv = np.meshgrid(x, y, z)
        self.voxels = (xv, yv, zv)
        print(self.voxels)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(xv, yv, zv, marker="o")
        plt.show()


env = Environment()
env.setBoundaries((-2, 2), (-2, 2), (0, 2))
env.setVoxelsResolution(20, 20, 10)
