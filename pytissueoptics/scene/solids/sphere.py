from pytissueoptics.scene.geometry import Vector, Triangle
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid
import itertools


class Sphere(Solid):
    """
        The Sphere is the 3D analog to the circle. Meshing a sphere requires an infinite number of vertices.
        The position refers to the vector from global origin to its centroid.
        The radius of the sphere will determine the outermost distance from its centroid.

        This class offers two possible methods to generate the sphere mesh.
        - With Quads: Specify the number of separation lines on the vertical axis and the horizontal axis of the sphere.
        - With Triangle: Specify the order of splitting. This will generate what is known as an IcoSphere.
    """

    def __init__(self,
                 radius: float,
                 order: int,
                 position: Vector = Vector(),
                 material: Material = Material(),
                 primitive: str = primitives.DEFAULT):
        surfaces = {'Sphere': []}
        self.radius = radius
        self.order = order
        self._primitive = primitive
        super().__init__(position=position, material=material, vertices=[], surfaces=surfaces)

    def _computeMesh(self):
        if self._primitive == primitives.TRIANGLE:
            self._computeTriangleMesh()
        else:
            raise NotImplementedError(f"Sphere mesh not implemented for primitive '{self._primitive}'")

    def _computeTriangleMesh(self):
        order = self.order

        phi = (1.0 + 5.0 ** (1 / 2)) / 2.0
        xyPlanePoints = [Vector(-1, phi, 0), Vector(1, phi, 0), Vector(-1, -phi, 0), Vector(1, -phi, 0)]
        yzPlanePoints = [Vector(0, -1, phi), Vector(0, 1, phi), Vector(0, -1, -phi), Vector(0, 1, -phi)]
        xzPlanePoints = [Vector(phi, 0, -1), Vector(phi, 0, 1), Vector(-phi, 0, -1), Vector(-phi, 0, 1)]
        self._vertices = list(itertools.chain(xyPlanePoints, yzPlanePoints, xzPlanePoints))
        V = self._vertices
        surfaces = [Triangle(V[0], V[11], V[5]),
                    Triangle(V[0], V[5], V[1]),
                    Triangle(V[0], V[1], V[7]),
                    Triangle(V[0], V[7], V[10]),
                    Triangle(V[0], V[10], V[11]),
                    Triangle(V[1], V[5], V[9]),
                    Triangle(V[5], V[11], V[4]),
                    Triangle(V[11], V[10], V[2]),
                    Triangle(V[10], V[7], V[6]),
                    Triangle(V[7], V[1], V[8]),
                    Triangle(V[3], V[9], V[4]),
                    Triangle(V[3], V[4], V[2]),
                    Triangle(V[3], V[2], V[6]),
                    Triangle(V[3], V[6], V[8]),
                    Triangle(V[3], V[8], V[9]),
                    Triangle(V[4], V[9], V[5]),
                    Triangle(V[2], V[4], V[11]),
                    Triangle(V[6], V[2], V[10]),
                    Triangle(V[8], V[6], V[7]),
                    Triangle(V[9], V[8], V[1])]


