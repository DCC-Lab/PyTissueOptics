import unittest

from pytissueoptics.scene import Vector
from pytissueoptics.scene.geometry import Triangle, primitives
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.solids.surfaceCollection import SurfaceCollection
from pytissueoptics.scene.viewer.mayavi import MayaviSolid


class TestMayaviSolid(unittest.TestCase):
    def createSimpleSolid(self) -> Solid:
        V = [Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0), Vector(1, 0, 0)]
        self.surfaces = SurfaceCollection()
        self.surfaces.add("Face", [Triangle(V[0], V[1], V[2]), Triangle(V[2], V[3], V[0])])
        self.vertices = V
        return Solid(position=Vector(0, 0, 0), vertices=self.vertices,
                     surfaces=self.surfaces, primitive=primitives.TRIANGLE)

    def testGivenNewMayaviSolid_shouldExtractMayaviMeshFromSolid(self):
        solid = self.createSimpleSolid()
        mayaviSolid = MayaviSolid(solid)

        mesh = mayaviSolid.mesh
        x, y, z, polygonIndices = mesh.components
        self.assertTrue(len(x) == len(y) == len(z))
        self.assertEqual([0, 0, 1, 1], x)
        self.assertEqual(len(solid.getPolygons()), len(polygonIndices))
        self.assertEqual((2, 3, 0), polygonIndices[1])
