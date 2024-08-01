import envtest
import unittest

from pytissueoptics.scene import Vector
from pytissueoptics.scene.geometry import Polygon, Vertex
from pytissueoptics.scene.shader import getSmoothNormal


class TestSmoothing(unittest.TestCase):
    def setUp(self):
        # Create a XY square polygon with normals pointing outwards along the Z axis.
        vertices = [Vertex(0, 0, 0), Vertex(1, 0, 0),
                    Vertex(1, 1, 0), Vertex(0, 1, 0)]
        normals = [Vector(-1, -1, 1), Vector(1, -1, 1),
                   Vector(1, 1, 1), Vector(-1, 1, 1)]
        for i in range(4):
            normals[i].normalize()
            vertices[i].normal = normals[i]

        self.initialNormal = Vector(0, 0, 1)
        self.polygon = Polygon(vertices=vertices, normal=self.initialNormal)
        self.polygon.toSmooth = True

    def testShouldReturnInterpolatedNormalAtDesiredPosition(self):
        position = Vector(0.5, 0.5, 0)
        smoothNormal = getSmoothNormal(self.polygon, position)
        self.assertEqual(Vector(0, 0, 1), smoothNormal)

    def testGivenPolygonIsNotToSmooth_shouldReturnDefaultPolygonNormal(self):
        self.polygon.toSmooth = False
        smoothNormal = getSmoothNormal(self.polygon, Vector(0.2, 0.2, 0))
        self.assertEqual(self.initialNormal, smoothNormal)

    def testGivenPositionOnSideOfPolygon_shouldReturnInterpolatedNormalBetweenSideVertices(self):
        position = Vector(0.5, 0, 0)
        smoothNormal = getSmoothNormal(self.polygon, position)
        expectedNormal = Vector(0, -1, 1)
        expectedNormal.normalize()
        self.assertEqual(expectedNormal, smoothNormal)

    def testGivenPositionOnVertex_shouldReturnVertexNormal(self):
        position = Vector(0, 0, 0)
        smoothNormal = getSmoothNormal(self.polygon, position)
        self.assertEqual(self.polygon.vertices[0].normal, smoothNormal)

if __name__ == "__main__":
    unittest.main()
