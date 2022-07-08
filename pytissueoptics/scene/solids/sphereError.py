from __future__ import division

from pytissueoptics import Ellipsoid, Sphere
from pytissueoptics.scene.geometry import Vertex
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


class SphereErrorVisualizer:
    def __init__(self):
        self._testSphere = None

    def _sampleSphereError(self, x, y, z):
        return np.array([1.0 - self._testSphere._radiusTowards(Vertex(x[i], y[i], z[i])) for i in range(len(x))])

    @staticmethod
    def _mplSphereSurface():
        u = np.linspace(0, 2 * np.pi, 1000)
        v = np.linspace(0, np.pi, 500)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    def displaySphereError(self, sphereObject: Ellipsoid):
        self._testSphere = sphereObject
        x, y, z = self._mplSphereSurface()
        radiusError = self._sampleSphereError(x, y, z)
        print(radiusError)

        norm = matplotlib.colors.SymLogNorm(1, vmin=radiusError.min(), vmax=radiusError.max())
        colors = plt.cm.viridis(norm(radiusError))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(x, y, z, facecolors=colors,
                               linewidth=0, antialiased=False)
        # sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        # sm.set_array(radiusError)
        # fig.colorbar(sm, shrink=0.5, aspect=5)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()


a = SphereErrorVisualizer()
a.displaySphereError(Sphere(radius=1, order=2))
