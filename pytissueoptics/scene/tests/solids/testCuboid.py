import unittest

from pytissueoptics.scene.geometry import Vector, primitives, Vertex
from pytissueoptics.scene.solids import Cuboid


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

    def testWhenStackOnNonExistentSurface_shouldNotStack(self):
        baseCuboid = Cuboid(4, 5, 3)
        otherCuboid = Cuboid(4, 5, 1)

        with self.assertRaises(Exception):
            baseCuboid.stack(otherCuboid, onSurface='BadSurfaceKey')

    def testWhenStackUnmatchedSurfaces_shouldNotStack(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        with self.assertRaises(Exception):
            baseCuboid.stack(otherCuboid, onSurface='right')

    def testWhenStackOnASurface_shouldMoveTheOtherCuboidToBeAdjacentToThisSurface(self):
        basePosition = Vector(2, 2, 1)
        otherPosition = Vector(5, 0, 4)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition)
        otherCuboid = Cuboid(5, 1, 4, position=otherPosition)

        baseCuboid.stack(otherCuboid, onSurface='bottom')

        self.assertEqual(baseCuboid.position + Vector(0, -2, 0), otherCuboid.position)

    def testWhenStack_shouldShareSurfacesWithTheOtherCuboid(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        baseCuboid.stack(otherCuboid, onSurface='top')

        self.assertEqual(baseCuboid.getPolygons('top'), otherCuboid.getPolygons('bottom'))

    def testWhenStack_shouldSetOtherCuboidEnvironmentAtInterface(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)
        topEnvironment = otherCuboid.getEnvironment()

        baseCuboid.stack(otherCuboid, onSurface='top')

        for polygon in baseCuboid.getPolygons('top'):
            self.assertEqual(topEnvironment, polygon.outsideEnvironment)

    def testWhenStack_shouldReturnANewCuboidMadeOfTheseTwoCuboids(self):
        basePosition = Vector(2, 2, 1)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition)
        otherCuboid = Cuboid(5, 1, 4)

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='bottom')

        self.assertEqual([5, 4, 4], cuboidStack.shape)
        self.assertEqual(basePosition - Vector(0, 0.5, 0), cuboidStack.position)

    def testWhenStack_shouldReturnANewCuboidWithAFirstInterface(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='top')

        self.assertTrue("interface0" in cuboidStack.surfaceLabels)

    def testWhenStack_shouldPreserveEnvironmentAtEachLayer(self):
        baseMaterial = "BaseMaterial"
        otherMaterial = "OtherMaterial"
        baseCuboid = Cuboid(5, 3, 4, material=baseMaterial, label="base")
        otherCuboid = Cuboid(5, 1, 4, material=otherMaterial, label="other")

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='top')

        interfacePolygon = cuboidStack.getPolygons("interface0")[0]

        self.assertEqual(baseMaterial, interfacePolygon.insideEnvironment.material)
        self.assertEqual(otherMaterial, interfacePolygon.outsideEnvironment.material)
        self.assertEqual(baseCuboid, interfacePolygon.insideEnvironment.solid)
        self.assertEqual(otherCuboid, interfacePolygon.outsideEnvironment.solid)

    def testWhenStackAnotherStack_shouldReturnANewCuboidWithAllStackInterfaces(self):
        baseCuboid1 = Cuboid(5, 3, 4)
        otherCuboid1 = Cuboid(5, 1, 4)
        cuboidStack1 = baseCuboid1.stack(otherCuboid1, onSurface='top')

        baseCuboid2 = Cuboid(2, 4, 4)
        otherCuboid2 = Cuboid(3, 4, 4)
        cuboidStack2 = baseCuboid2.stack(otherCuboid2, onSurface='right')

        cuboidStack = cuboidStack1.stack(cuboidStack2, onSurface='right')

        for i in range(3):
            self.assertTrue(f"interface{i}" in cuboidStack.surfaceLabels)

    def testWhenContainsWithVerticesThatAreAllInsideTheCuboid_shouldReturnTrue(self):
        cuboid = Cuboid(1, 1, 8, position=Vector(2, 2, 0))
        cuboid.rotate(30, 0, 0)
        vertices = [Vertex(2, 2 - 1.71, 3), Vertex(2, 2, 0)]

        self.assertTrue(cuboid.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotAllInsideTheCuboid_shouldReturnFalse(self):
        cuboid = Cuboid(1, 1, 8, position=Vector(2, 2, 0))
        cuboid.rotate(30, 180, 0)
        vertices = [Vertex(2, 2, 3), Vertex(2, 2, 0)]

        self.assertFalse(cuboid.contains(*vertices))
