import unittest

from pytissueoptics.scene.geometry import Vector, primitives
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

        self.assertEqual(self._getSurfaceCentroid(cuboid, "Left"), Vector(0, 1, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "Right"), Vector(2, 1, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "Bottom"), Vector(1, 0, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "Top"), Vector(1, 2, 1))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "Front"), Vector(1, 1, 0))
        self.assertEqual(self._getSurfaceCentroid(cuboid, "Back"), Vector(1, 1, 2))

    @staticmethod
    def _getSurfaceCentroid(cuboid, surfaceName: str) -> Vector:
        centroid = Vector(0, 0, 0)
        surfacePolygons = cuboid.getPolygons(surfaceName)
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
            baseCuboid.stack(otherCuboid, onSurface='Right')

    def testWhenStackOnASurface_shouldMoveTheOtherCuboidToBeAdjacentToThisSurface(self):
        basePosition = Vector(2, 2, 1)
        otherPosition = Vector(5, 0, 4)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition)
        otherCuboid = Cuboid(5, 1, 4, position=otherPosition)

        baseCuboid.stack(otherCuboid, onSurface='Bottom')

        self.assertEqual(baseCuboid.position + Vector(0, -2, 0), otherCuboid.position)

    def testWhenStack_shouldShareSurfacesWithTheOtherCuboid(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        baseCuboid.stack(otherCuboid, onSurface='Top')

        self.assertEqual(baseCuboid.getPolygons('Top'), otherCuboid.getPolygons('Bottom'))

    def testWhenStack_shouldSetOtherCuboidMaterialAtInterface(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        baseCuboid.stack(otherCuboid, onSurface='Top')

        for polygon in baseCuboid.getPolygons('Top'):
            self.assertEqual(polygon.outsideMaterial, otherCuboid.getMaterial())

    def testWhenStack_shouldReturnANewCuboidMadeOfTheseTwoCuboids(self):
        basePosition = Vector(2, 2, 1)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition)
        otherCuboid = Cuboid(5, 1, 4)

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='Top')

        self.assertEqual([5, 4, 4], cuboidStack.shape)
        self.assertEqual(basePosition + Vector(0, 0.5, 0), cuboidStack.position)

    def testWhenStack_shouldReturnANewCuboidWithAFirstInterface(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='Top')

        self.assertTrue("Interface0" in cuboidStack.surfaceNames)

    def testWhenStackAnotherStack_shouldReturnANewCuboidWithAllStackInterfaces(self):
        baseCuboid1 = Cuboid(5, 3, 4)
        otherCuboid1 = Cuboid(5, 1, 4)
        cuboidStack1 = baseCuboid1.stack(otherCuboid1, onSurface='Top')

        baseCuboid2 = Cuboid(2, 4, 4)
        otherCuboid2 = Cuboid(3, 4, 4)
        cuboidStack2 = baseCuboid2.stack(otherCuboid2, onSurface='Right')

        cuboidStack = cuboidStack1.stack(cuboidStack2, onSurface='Right')

        for i in range(3):
            self.assertTrue(f"Interface{i}" in cuboidStack.surfaceNames)

    def testWhenContainsWithVerticesThatAreAllInsideTheCuboid_shouldReturnTrue(self):
        cuboid = Cuboid(1, 1, 8, position=Vector(2, 2, 0))
        cuboid.rotate(30, 0, 0)
        vertices = [Vector(2, 2 - 1.71, 3), Vector(2, 2, 0)]

        self.assertTrue(cuboid.contains(*vertices))

    def testWhenContainsWithVerticesThatAreNotAllInsideTheCuboid_shouldReturnFalse(self):
        cuboid = Cuboid(1, 1, 8, position=Vector(2, 2, 0))
        cuboid.rotate(30, 180, 0)
        vertices = [Vector(2, 2, 3), Vector(2, 2, 0)]

        self.assertFalse(cuboid.contains(*vertices))
