import unittest
from unittest.mock import patch, Mock

from pytissueoptics.scene.geometry import Vector, Quad
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid


class TestSolid(unittest.TestCase):
    def setUp(self):
        self.CUBOID_VERTICES = [Vector(-1, -1, -1), Vector(1, -1, -1),
                                Vector(1, 1, -1), Vector(-1, 1, -1),
                                Vector(-1, -1, 1), Vector(1, -1, 1),
                                Vector(1, 1, 1), Vector(-1, 1, 1)]
        V = self.CUBOID_VERTICES
        self.CUBOID_SURFACES = {'Front': [Quad(V[0], V[1], V[2], V[3])], 'Back': [Quad(V[5], V[4], V[7], V[6])],
                                'Left': [Quad(V[4], V[0], V[3], V[7])], 'Right': [Quad(V[1], V[5], V[6], V[2])],
                                'Top': [Quad(V[3], V[2], V[6], V[7])], 'Bottom': [Quad(V[4], V[5], V[1], V[0])]}

        self.material = Material()
        self.position = Vector(2, 2, 0)
        self.solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                           surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)

    def testShouldBeAtDesiredPosition(self):
        self.assertEqual(self.position, self.solid.position)
        self.assertEqual(Vector(-1, -1, -1) + self.position, self.CUBOID_VERTICES[0])

    def testShouldSetInsideMaterial(self):
        surfaceElement = self.CUBOID_SURFACES['Front'][0]
        self.assertEqual(self.material, surfaceElement.insideMaterial)

    def testWhenTranslateTo_shouldTranslateToThisNewPosition(self):
        newPosition = Vector(0, 0, 0)

        self.solid.translateTo(newPosition)

        self.assertEqual(newPosition, self.solid.position)
        self.assertEqual(Vector(-1, -1, -1) + newPosition, self.CUBOID_VERTICES[0])

    def testWhenTranslateBy_shouldTranslateByTheDesiredAmount(self):
        initialY = self.solid.position.y
        aTranslation = Vector(0, 2, 0)

        self.solid.translateBy(aTranslation)

        self.assertEqual(initialY + aTranslation.y, self.solid.position.y)

    def testWhenRotate_shouldRotateVertices(self):
        self.solid.rotate(xTheta=90, yTheta=90, zTheta=90)

        expectedRotatedVertex = Vector(-1, -1, 1) + self.position
        self.assertAlmostEqual(expectedRotatedVertex.x, self.CUBOID_VERTICES[0].x)
        self.assertAlmostEqual(expectedRotatedVertex.y, self.CUBOID_VERTICES[0].y)
        self.assertAlmostEqual(expectedRotatedVertex.z, self.CUBOID_VERTICES[0].z)

    @patch('pytissueoptics.scene.solids.Solid._computeMesh')
    def testWhenRotate_shouldRotateSurfaces(self, fakeComputeMesh):
        quadMock = Mock()
        self.CUBOID_SURFACES['Front'] = [quadMock]
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)
        surfaceResetNormalCount = quadMock.resetNormal.call_count

        solid.rotate(xTheta=90, yTheta=90, zTheta=90)

        self.assertEqual(surfaceResetNormalCount + 1, quadMock.resetNormal.call_count)
