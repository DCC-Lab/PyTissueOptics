import unittest

from pytissueoptics.scene import Vector
from pytissueoptics.scene.geometry import Polygon, Quad, SurfaceCollection, Triangle, primitives
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.viewer.mayavi import MayaviSolid


class TestMayaviSolid(unittest.TestCase):
    def createSimpleSolid(self, primitive=primitives.TRIANGLE) -> Solid:
        V = [Vector(0, 0, 0), Vector(0, 1, 0), Vector(1, 1, 0), Vector(1, 0, 0), Vector(0.5, -1, 0)]
        self.surfaces = SurfaceCollection()
        if primitive == primitives.TRIANGLE:
            self.surfaces.add("Face", [Triangle(V[0], V[1], V[2]), Triangle(V[0], V[2], V[3])])
        if primitive == primitives.QUAD:
            self.surfaces.add("Face", [Quad(V[0], V[1], V[2], V[3])])
        if primitive == "Polygon":
            self.surfaces.add("Face", [Polygon([V[0], V[1], V[2], V[3], V[4]]), Triangle(V[0], V[1], V[2])])
        self.vertices = V
        return Solid(position=Vector(0, 0, 0), vertices=self.vertices, surfaces=self.surfaces, primitive=primitive)

    def testGivenNewMayaviSolidWithTrianglePrimitive_shouldExtractMayaviTriangleMeshFromSolid(self):
        solid = self.createSimpleSolid()
        mayaviSolid = MayaviSolid(solid)

        x, y, z, polygonIndices = mayaviSolid.triangleMesh.components
        self.assertTrue(len(x) == len(y) == len(z))
        self.assertEqual(len(solid.getPolygons()), len(polygonIndices))
        self.assertEqual((0, 2, 3), polygonIndices[1])

    def testGivenNewMayaviSolidWithQuadPrimitive_shouldExtractMayaviTriangleMeshFromSolid(self):
        solid = self.createSimpleSolid(primitive=primitives.QUAD)
        mayaviSolid = MayaviSolid(solid)

        x, y, z, polygonIndices = mayaviSolid.triangleMesh.components
        self.assertTrue(len(x) == len(y) == len(z))
        self.assertEqual(2 * len(solid.getPolygons()), len(polygonIndices))
        self.assertEqual((0, 2, 3), polygonIndices[1])

    def testGivenNewMayaviSolidWithArbitraryPolygonPrimitives_shouldExtractMayaviTriangleMeshFromSolid(self):
        solid = self.createSimpleSolid(primitive="Polygon")
        mayaviSolid = MayaviSolid(solid)

        x, y, z, polygonIndices = mayaviSolid.triangleMesh.components
        self.assertTrue(len(x) == len(y) == len(z))
        self.assertEqual(4, len(polygonIndices))
        self.assertEqual((0, 2, 3), polygonIndices[1])

    def testGivenNewMayaviSolidWithLoadNormals_shouldExtractMayaviNormalsFromSolid(self):
        solid = self.createSimpleSolid()
        mayaviSolid = MayaviSolid(solid, loadNormals=True)

        x, y, z, u, v, w = mayaviSolid.normals.components
        self.assertEqual(len(solid.getPolygons()), len(x))
        self.assertTrue(len(x) == len(y) == len(z) == len(u) == len(v) == len(w))

        for i, polygon in enumerate(self.surfaces.getPolygons()):
            self.assertEqual([x[i], y[i], z[i]], polygon.getCentroid().array)
            self.assertEqual([u[i], v[i], w[i]], polygon.normal.array)
