import unittest

import numpy as np

from pytissueoptics.scene.geometry import Vector, primitives, Vertex, INTERFACE_KEY
from pytissueoptics.scene.solids import Cuboid, Cube


class TestCuboid(unittest.TestCase):
    def testGivenANewDefaultCuboid_shouldBePlacedAtOrigin(self):
        cuboid = Cuboid(8, 1, 3)
        self.assertEqual(Vector(0, 0, 0), cuboid.position)

    def testGivenANewCuboid_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        cuboid = Cuboid(8, 1, 3, position=position)
        self.assertEqual(position, cuboid.position)

    def testGivenACuboidWithTrianglePrimitive_shouldHaveASurfaceGroupForEachCuboidFace(self):
        return self._testGivenACuboidWithAnyPrimitive_shouldHaveASurfaceGroupForEachCuboidFace(primitives.TRIANGLE)

    def testGivenACuboidWithQuadPrimitive_shouldHaveASurfaceGroupForEachCuboidFace(self):
        return self._testGivenACuboidWithAnyPrimitive_shouldHaveASurfaceGroupForEachCuboidFace(primitives.QUAD)

    def _testGivenACuboidWithAnyPrimitive_shouldHaveASurfaceGroupForEachCuboidFace(self, anyPrimitive):
        position = Vector(1, 1, 1)
        cuboid = Cuboid(2, 2, 2, position=position, primitive=anyPrimitive)

        self.assertEqual(self._getSurfaceCentroid(cuboid, "left"), Vector(0, 1, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "right"), Vector(2, 1, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "bottom"), Vector(1, 0, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "top"), Vector(1, 2, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "front"), Vector(1, 1, 0))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "back"), Vector(1, 1, 2))

    @staticmethod
    def _getSurfaceCentroid(cuboid, surfaceLabel: str) -> Vector:
        centroid = Vector(0, 0, 0)
        surfacePolygons = cuboid.getPolygons(surfaceLabel)
        for polygon in surfacePolygons:
            centroid += polygon.getCentroid()
        centroid.divide(len(surfacePolygons))
        return centroid

    def testWhenStackCuboidsWithTheSameLabel_shouldNotStack(self):
        baseCuboid = Cuboid(4, 5, 3)
        otherCuboid = Cuboid(4, 5, 1)

        with self.assertRaises(AssertionError):
            baseCuboid.stack(otherCuboid, 'front')

    def testWhenStackOnNonExistentSurface_shouldNotStack(self):
        baseCuboid = Cuboid(4, 5, 3, label="BaseCuboid")
        otherCuboid = Cuboid(4, 5, 1, label="OtherCuboid")

        with self.assertRaises(Exception):
            baseCuboid.stack(otherCuboid, onSurface='BadSurfaceKey')

    def testWhenStackUnmatchedSurfaces_shouldNotStack(self):
        baseCuboid = Cuboid(5, 3, 4, label="BaseCuboid")
        otherCuboid = Cuboid(5, 1, 4, label="OtherCuboid")

        with self.assertRaises(Exception):
            baseCuboid.stack(otherCuboid, onSurface='right')

    def testWhenStackOnASurface_shouldMoveTheOtherCuboidToBeAdjacentToThisSurface(self):
        basePosition = Vector(2, 2, 1)
        otherPosition = Vector(5, 0, 4)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition, label="BaseCuboid")
        otherCuboid = Cuboid(5, 1, 4, position=otherPosition, label="OtherCuboid")

        baseCuboid.stack(otherCuboid, onSurface='bottom')

        self.assertEqual(baseCuboid.position + Vector(0, -2, 0), otherCuboid.position)

    def testWhenStack_shouldShareSurfacesWithTheOtherCuboid(self):
        baseCuboid = Cuboid(5, 3, 4, label="BaseCuboid")
        otherCuboid = Cuboid(5, 1, 4, label="OtherCuboid")

        baseCuboid.stack(otherCuboid, onSurface='top')

        self.assertEqual(baseCuboid.getPolygons('top'), otherCuboid.getPolygons('bottom'))

    def testWhenStack_shouldSetOtherCuboidEnvironmentAtInterface(self):
        baseCuboid = Cuboid(5, 3, 4, label="BaseCuboid")
        otherCuboid = Cuboid(5, 1, 4, label="OtherCuboid")
        topEnvironment = otherCuboid.getEnvironment()

        baseCuboid.stack(otherCuboid, onSurface='top')

        for polygon in baseCuboid.getPolygons('top'):
            self.assertEqual(topEnvironment, polygon.outsideEnvironment)

    def testWhenStack_shouldReturnANewCuboidMadeOfTheseTwoCuboids(self):
        basePosition = Vector(2, 2, 1)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition, label="BaseCuboid")
        otherCuboid = Cuboid(5, 1, 4, label="OtherCuboid")

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='bottom')

        self.assertEqual([5, 4, 4], cuboidStack.shape)
        self.assertEqual(basePosition - Vector(0, 0.5, 0), cuboidStack.position)

    def testWhenStack_shouldReturnANewCuboidWithAFirstInterface(self):
        baseCuboid = Cuboid(5, 3, 4, label="BaseCuboid")
        otherCuboid = Cuboid(5, 1, 4, label="OtherCuboid")

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='top')

        self.assertTrue(f"{INTERFACE_KEY}0" in cuboidStack.surfaceLabels)

    def testWhenStack_shouldPreserveEnvironmentAtEachLayer(self):
        baseMaterial = "BaseMaterial"
        otherMaterial = "OtherMaterial"
        baseCuboid = Cuboid(5, 3, 4, material=baseMaterial, label="base")
        otherCuboid = Cuboid(5, 1, 4, material=otherMaterial, label="other")

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='top')

        interfacePolygon = cuboidStack.getPolygons(f"{INTERFACE_KEY}0")[0]

        self.assertEqual(baseMaterial, interfacePolygon.insideEnvironment.material)
        self.assertEqual(otherMaterial, interfacePolygon.outsideEnvironment.material)
        self.assertEqual(baseCuboid, interfacePolygon.insideEnvironment.solid)
        self.assertEqual(otherCuboid, interfacePolygon.outsideEnvironment.solid)

    def testWhenStackAnotherStack_shouldReturnANewCuboidWithAllStackInterfaces(self):
        baseCuboid1 = Cuboid(5, 3, 4, label="BaseCuboid1")
        otherCuboid1 = Cuboid(5, 1, 4, label="OtherCuboid1")
        cuboidStack1 = baseCuboid1.stack(otherCuboid1, onSurface='top')

        baseCuboid2 = Cuboid(2, 4, 4, label="BaseCuboid2")
        otherCuboid2 = Cuboid(3, 4, 4, label="OtherCuboid2")
        cuboidStack2 = baseCuboid2.stack(otherCuboid2, onSurface='right')

        cuboidStack = cuboidStack1.stack(cuboidStack2, onSurface='right')

        for i in range(3):
            self.assertTrue(f"{INTERFACE_KEY}{i}" in cuboidStack.surfaceLabels)

    def testWhenStackAnotherStackNotAlongTheAlreadyStackedAxis_shouldNotStack(self):
        baseCuboid1 = Cuboid(5, 3, 4, label="base1")
        otherCuboid1 = Cuboid(5, 1, 4, label="other1")
        cuboidStack1 = baseCuboid1.stack(otherCuboid1, onSurface='top')

        baseCuboid2 = Cuboid(2, 4, 4, label="base2")
        otherCuboid2 = Cuboid(3, 4, 4, label="other2")
        cuboidStack2 = baseCuboid2.stack(otherCuboid2, onSurface='right')

        with self.assertRaises(Exception):
            cuboidStack1.stack(cuboidStack2, onSurface='top')

    def testWhenContainsWithVerticesThatAreAllInsideTheCuboid_shouldReturnTrue(self):
        cuboid = Cuboid(1, 1, 8, position=Vector(2, 2, 0))
        cuboid.rotate(45, 0, 0)
        vertices = [Vertex(2, 2 - 2.5, 2.5), Vertex(2, 2, 0)]

        self.assertTrue(cuboid.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotAllInsideTheCuboid_shouldReturnFalse(self):
        cuboid = Cuboid(1, 1, 8, position=Vector(2, 2, 0))
        cuboid.rotate(30, 180, 0)
        vertices = [Vertex(2, 2, 3), Vertex(2, 2, 0)]

        self.assertFalse(cuboid.contains(*vertices))

    def testGivenTwoCubesContainedAfterDifferentRotationCenters_whenContains_shouldReturnTrue(self):
        outerCube = Cube(3, label="outerCube")
        innerCube = Cube(2.9, label="innerCube")
        self.assertTrue(outerCube.contains(*innerCube.vertices))

        distanceFromRotationCenter = 10
        rotationCenterXY = -np.sin(np.pi/4) * distanceFromRotationCenter
        rotationCenter = Vector(rotationCenterXY, rotationCenterXY, 0)

        innerCube.translateTo(Vector(rotationCenterXY+distanceFromRotationCenter, rotationCenterXY, 0))
        outerCube.rotate(0, 0, 45)
        innerCube.rotate(0, 0, 45, rotationCenter)

        self.assertTrue(outerCube.contains(*innerCube.vertices))
