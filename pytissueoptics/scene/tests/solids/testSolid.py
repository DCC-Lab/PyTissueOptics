import unittest

from mockito import mock, verify, when

from pytissueoptics.scene.geometry import Vector, Quad, Polygon
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import SurfaceCollection


class TestSolid(unittest.TestCase):
    def setUp(self):
        self.CUBOID_VERTICES = [Vector(-1, -1, -1), Vector(1, -1, -1),
                                Vector(1, 1, -1), Vector(-1, 1, -1),
                                Vector(-1, -1, 1), Vector(1, -1, 1),
                                Vector(1, 1, 1), Vector(-1, 1, 1)]
        V = self.CUBOID_VERTICES
        self.CUBOID_SURFACES = SurfaceCollection()
        self.CUBOID_SURFACES.add('Front', [Quad(V[0], V[1], V[2], V[3])])
        self.CUBOID_SURFACES.add('Back', [Quad(V[5], V[4], V[7], V[6])])
        self.CUBOID_SURFACES.add('Left', [Quad(V[4], V[0], V[3], V[7])])
        self.CUBOID_SURFACES.add('Right', [Quad(V[1], V[5], V[6], V[2])])
        self.CUBOID_SURFACES.add('Top', [Quad(V[3], V[2], V[6], V[7])])
        self.CUBOID_SURFACES.add('Bottom', [Quad(V[4], V[5], V[1], V[0])])

        self.material = Material()
        self.position = Vector(2, 2, 0)
        self.solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                           surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)

    def testShouldBeAtDesiredPosition(self):
        self.assertEqual(self.position, self.solid.position)
        self.assertEqual(Vector(-1, -1, -1) + self.position, self.CUBOID_VERTICES[0])

    def testShouldSetInsideMaterialOfAllItsSurfaces(self):
        self.assertEqual(self.material, self.solid.getMaterial())
        self.assertEqual(self.material, self.solid.getMaterial("Top"))

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

    def testWhenRotate_shouldRotateItsVertices(self):
        expectedRotatedVertex0 = Vector(-1, -1, 1) + self.position
        expectedRotatedVertex5 = Vector(1, -1, -1) + self.position

        self.solid.rotate(xTheta=90, yTheta=90, zTheta=90)

        self.assertAlmostEqual(expectedRotatedVertex0.x, self.CUBOID_VERTICES[0].x)
        self.assertAlmostEqual(expectedRotatedVertex0.y, self.CUBOID_VERTICES[0].y)
        self.assertAlmostEqual(expectedRotatedVertex0.z, self.CUBOID_VERTICES[0].z)

        self.assertAlmostEqual(expectedRotatedVertex5.x, self.CUBOID_VERTICES[5].x)
        self.assertAlmostEqual(expectedRotatedVertex5.y, self.CUBOID_VERTICES[5].y)
        self.assertAlmostEqual(expectedRotatedVertex5.z, self.CUBOID_VERTICES[5].z)

    def testWhenRotate_shouldRotateItsPolygons(self):
        polygon = mock(Polygon)
        when(polygon).resetNormal().thenReturn()
        when(polygon).setInsideMaterial(...).thenReturn()
        self.CUBOID_SURFACES.setPolygons('Front', [polygon])
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)

        solid.rotate(xTheta=90, yTheta=90, zTheta=90)

        verify(polygon, times=1).resetNormal()

    def testWhenRotate_shouldChangeBboxOfSolidAndPolygons(self):
        polygon = mock(Polygon)
        when(polygon).resetNormal().thenReturn()
        when(polygon).resetBoundingBox().thenReturn()
        when(polygon).setInsideMaterial(...).thenReturn()

        self.CUBOID_SURFACES.setPolygons('Front', [polygon])
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)
        oldBbox = solid.bbox
        solid.rotate(xTheta=90, yTheta=90, zTheta=90)
        newBbox = solid.bbox

        # once during the __init__, once during the positioning, once during the rotation = 3
        verify(polygon, times=3).resetBoundingBox()
        self.assertNotEqual(oldBbox, newBbox)


    def testWhenTranslate_shouldChangeBboxOfSolidAndPolygons(self):
        polygon = mock(Polygon)
        when(polygon).resetNormal().thenReturn()
        when(polygon).resetBoundingBox().thenReturn()
        when(polygon).setInsideMaterial(...).thenReturn()

        self.CUBOID_SURFACES.setPolygons('Front', [polygon])
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)
        oldBbox = solid.bbox
        solid.translateTo(Vector(1, -1, -1))
        newBbox = solid.bbox

        # once during the __init__, once during the positioning, once during the translation = 3
        verify(polygon, times=3).resetBoundingBox()
        self.assertNotEqual(oldBbox, newBbox)