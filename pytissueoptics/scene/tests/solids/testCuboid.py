import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cuboid


class TestCuboid(unittest.TestCase):
    def testGivenANewDefaultCuboid_shouldBePlacedAtOrigin(self):
        cuboid = Cuboid(8, 1, 3)
        self.assertEqual(Vector(0, 0, 0), cuboid.position)

    def testGivenANewCuboid_shouldBePlacedAtDesiredPosition(self):
        position = Vector(2, 2, 1)
        cuboid = Cuboid(8, 1, 3, position=position)
        self.assertEqual(position, cuboid.position)

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

        baseCuboid.stack(otherCuboid, onSurface='Top')

        self.assertEqual(baseCuboid.position + Vector(0, 2, 0), otherCuboid.position)

    def testWhenStack_shouldShareSurfacesWithTheOtherCuboid(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        baseCuboid.stack(otherCuboid, onSurface='Top')

        self.assertEqual(baseCuboid._surfaces['Top'], otherCuboid._surfaces['Bottom'])

    def testWhenStack_shouldSetOtherCuboidMaterialAtInterface(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        baseCuboid.stack(otherCuboid, onSurface='Top')

        for surface in baseCuboid._surfaces['Top']:
            self.assertEqual(surface.outsideMaterial, otherCuboid._material)

    def testWhenStack_shouldReturnANewSolidMadeOfTheseTwoCuboids(self):
        basePosition = Vector(2, 2, 1)
        baseCuboid = Cuboid(5, 3, 4, position=basePosition)
        otherCuboid = Cuboid(5, 1, 4)

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='Top')
    
        self.assertEqual(basePosition + Vector(0, 1, 0), cuboidStack.position)

    def testWhenStack_shouldReturnANewSolidWithAFirstInterface(self):
        baseCuboid = Cuboid(5, 3, 4)
        otherCuboid = Cuboid(5, 1, 4)

        cuboidStack = baseCuboid.stack(otherCuboid, onSurface='Top')

        self.assertTrue("Interface0" in cuboidStack._surfaces)
        # todo: surfaces should be in a Container object to which we can ask for face group keys
