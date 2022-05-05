import unittest

from pytissueoptics.scene.geometry import Vector, Vertex
from pytissueoptics.scene.solids import SolidGroup, Cube


class TestSolid(unittest.TestCase):
    def setUp(self):
        self.material = "A Material"
        self.position = Vector(0, 0, 0)
        cuboid1 = Cube(edge=2, position=Vector(2, 0, 0), material=self.material, label="cuboid1")
        cuboid2 = Cube(edge=2, position=Vector(-2, 0, 0), material=self.material, label="cuboid2")
        self.solidGroup = SolidGroup([cuboid1, cuboid2])

    def testShouldBeAtDesiredPosition(self):
        self.assertEqual(self.position, self.solidGroup.position)
        self.assertEqual(Vector(2, 0, 0) + self.position, self.solidGroup.getSolids()[0].position)

    def testWhenTranslateTo_shouldTranslateToThisNewPosition(self):
        newPosition = Vector(1, 1, 1)

        self.solidGroup.translateTo(newPosition)

        self.assertEqual(newPosition, self.solidGroup.position)
        self.assertEqual(Vector(2, 0, 0) + newPosition, self.solidGroup.getSolids()[0].position)

    def testWhenTranslateBy_shouldTranslateByTheDesiredAmount(self):
        initialY = self.solidGroup.position.y
        aTranslation = Vector(0, 2, 0)

        self.solidGroup.translateBy(aTranslation)

        self.assertEqual(initialY + aTranslation.y, self.solidGroup.position.y)
        self.assertEqual(Vector(2, 0, 0) + aTranslation, self.solidGroup.getSolids()[0].position)

    def testWhenRotate_shouldRotateSolidsPositionsAndVertices(self):
        expectedRotatedVertex0to5 = Vertex(x=-1, y=-1, z=1)
        expectedRotatedCuboid1Position = Vertex(x=-2, y=0, z=0)

        self.solidGroup.rotate(xTheta=0, yTheta=180, zTheta=0)

        self.assertAlmostEqual(expectedRotatedCuboid1Position.x, self.solidGroup.getSolids()[0].position.x)
        self.assertAlmostEqual(expectedRotatedCuboid1Position.y, self.solidGroup.getSolids()[0].position.y)
        self.assertAlmostEqual(expectedRotatedCuboid1Position.z, self.solidGroup.getSolids()[0].position.z)

        self.assertAlmostEqual(expectedRotatedVertex0to5.x, self.solidGroup.getSolids()[0].getVertices()[4].x)
        self.assertAlmostEqual(expectedRotatedVertex0to5.y, self.solidGroup.getSolids()[0].getVertices()[4].y)
        self.assertAlmostEqual(expectedRotatedVertex0to5.z, self.solidGroup.getSolids()[0].getVertices()[4].z)

    def testWhenRotate_shouldRotateBBox(self):
        oldBbox = self.solidGroup.getBoundingBox()
        self.solidGroup.rotate(xTheta=0, yTheta=90, zTheta=0)
        self.assertNotAlmostEqual(oldBbox.xLim[0], self.solidGroup.getBoundingBox().xLim[0])
        self.assertNotAlmostEqual(oldBbox.xLim[1], self.solidGroup.getBoundingBox().xLim[1])
        self.assertNotAlmostEqual(oldBbox.zLim[0], self.solidGroup.getBoundingBox().zLim[0])
        self.assertNotAlmostEqual(oldBbox.zLim[1], self.solidGroup.getBoundingBox().zLim[1])
        self.solidGroup.rotate(xTheta=0, yTheta=90, zTheta=0)
        self.assertAlmostEqual(oldBbox.xLim[0], self.solidGroup.getBoundingBox().xLim[0])
        self.assertAlmostEqual(oldBbox.xLim[1], self.solidGroup.getBoundingBox().xLim[1])
        self.assertAlmostEqual(oldBbox.yLim[0], self.solidGroup.getBoundingBox().yLim[0])
        self.assertAlmostEqual(oldBbox.yLim[1], self.solidGroup.getBoundingBox().yLim[1])
        self.assertAlmostEqual(oldBbox.zLim[0], self.solidGroup.getBoundingBox().zLim[0])
        self.assertAlmostEqual(oldBbox.zLim[1], self.solidGroup.getBoundingBox().zLim[1])
