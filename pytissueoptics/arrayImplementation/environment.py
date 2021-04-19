from typing import NamedTuple
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from arrayImplementation.sources import IsotropicSource

class Components(NamedTuple):
    sources: list
    geometries: list


class Environment:
    def __init__(self):
        self.voxels = None
        self.boundaries = None
        self.source = None
        self.components = []

    def place(self, component, position, label):
        self.components.append((component, position, label))

    def setBoundaries(self, x, y, z):
        self.boundaries = [x, y, z]

    def setVoxelsResolution(self, x, y, z):
        x = np.linspace(self.boundaries[0][0], self.boundaries[0][1], x)
        y = np.linspace(self.boundaries[1][0], self.boundaries[1][1], y)
        z = np.linspace(self.boundaries[2][0], self.boundaries[2][1], z)
        xv, yv, zv = np.meshgrid(x, y, z)
        self.voxels = (xv, yv, zv)

    def viewVoxels(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        xv, yv, zv = self.voxels
        ax.scatter3D(xv, yv, zv, marker="o")
        plt.show()

    def findNearestPoint(self, positions):
        xv, yv, zv = self.voxels
        ux = positions[:, 0]
        uy = positions[:, 1]
        uz = positions[:, 2]
        d = (ux-xv)**2 + (uy-yv)**2 + (uz-zv)**2
        print(d)


env = Environment()
env.setBoundaries((-2, 2), (-2, 2), (0, 2))
env.setVoxelsResolution(20, 20, 10)
env.viewVoxels()

source = IsotropicSource(100)
photons = source.initializePhotons()
env.findNearestPoint(photons.r)
