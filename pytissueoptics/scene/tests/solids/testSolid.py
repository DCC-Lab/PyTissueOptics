import math
import unittest

from mockito import mock, verify, when

from pytissueoptics.scene.geometry import Vector, Quad, Polygon, Vertex, Environment
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import SurfaceCollection, INTERFACE_KEY


class TestSolid(unittest.TestCase):
    def setUp(self):
        self.CUBOID_VERTICES = [Vertex(-1, -1, -1), Vertex(1, -1, -1),
                                Vertex(1, 1, -1), Vertex(-1, 1, -1),
                                Vertex(-1, -1, 1), Vertex(1, -1, 1),
                                Vertex(1, 1, 1), Vertex(-1, 1, 1)]
        V = self.CUBOID_VERTICES
        self.CUBOID_SURFACES = SurfaceCollection()
        self.CUBOID_SURFACES.add('front', [Quad(V[0], V[1], V[2], V[3])])
        self.CUBOID_SURFACES.add('back', [Quad(V[5], V[4], V[7], V[6])])
        self.CUBOID_SURFACES.add('left', [Quad(V[4], V[0], V[3], V[7])])
        self.CUBOID_SURFACES.add('right', [Quad(V[1], V[5], V[6], V[2])])
        self.CUBOID_SURFACES.add('top', [Quad(V[3], V[2], V[6], V[7])])
        self.CUBOID_SURFACES.add('bottom', [Quad(V[4], V[5], V[1], V[0])])

        self.material = "A Material"
        self.position = Vector(2, 2, 0)
        self.solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                           surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)

    def testShouldBeAtDesiredPosition(self):
        self.assertEqual(self.position, self.solid.position)
        self.assertEqual(Vector(-1, -1, -1) + self.position, self.CUBOID_VERTICES[0])

    def testShouldSetInsideEnvironmentOfAllItsSurfaces(self):
        environment = Environment(self.material, self.solid)
        self.assertEqual(environment, self.solid.getEnvironment())
        self.assertEqual(environment, self.solid.getEnvironment("top"))

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
        polygon = self.createPolygonMock()
        self.CUBOID_SURFACES.setPolygons('front', [polygon])
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)

        solid.rotate(xTheta=90, yTheta=90, zTheta=90)

        verify(polygon, times=1).resetNormal()

    def testWhenRotate_shouldRotateBBoxOfSolidAndPolygons(self):
        polygon = self.createPolygonMock()
        self.CUBOID_SURFACES.setPolygons('front', [polygon])
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)
        oldBbox = solid.bbox

        solid.rotate(xTheta=90, yTheta=90, zTheta=90)

        # once during the __init__, once during the positioning, once during the rotation = 3
        verify(polygon, times=3).resetBoundingBox()
        self.assertNotEqual(oldBbox, solid.bbox)

    def testWhenScale_shouldScaleAllVerticesFromTheCenter(self):
        self.solid.translateTo(Vector(5, 5, 5))

        self.solid.scale(5)

        self.assertEqual(Vector(5, 5, 5), self.solid.position)
        self.assertEqual(Vector(0, 0, 0), self.solid.vertices[0])
        self.assertEqual(Vector(10, 10, 10), self.solid.vertices[6])

    def testWhenRotateAroundAnotherCenterOfRotation_shouldRotateCorrectlyVertices(self):
        expectedRotatedVertex0 = Vector(-1, -1, -1) + self.position
        expectedRotatedVertex5 = Vector(1, -3, -1) + self.position

        self.solid.rotate(xTheta=90, yTheta=0, zTheta=0, rotationCenter=Vector(1, 1, -1))

        self.assertAlmostEqual(expectedRotatedVertex0.x, self.CUBOID_VERTICES[0].x)
        self.assertAlmostEqual(expectedRotatedVertex0.y, self.CUBOID_VERTICES[0].y)
        self.assertAlmostEqual(expectedRotatedVertex0.z, self.CUBOID_VERTICES[0].z)

        self.assertAlmostEqual(expectedRotatedVertex5.x, self.CUBOID_VERTICES[5].x)
        self.assertAlmostEqual(expectedRotatedVertex5.y, self.CUBOID_VERTICES[5].y)
        self.assertAlmostEqual(expectedRotatedVertex5.z, self.CUBOID_VERTICES[5].z)

    def testWhenRotateAroundAnotherCenterOfRotation_shouldRotatePositionOfSolid(self):
        self.solid.rotate(xTheta=90, yTheta=0, zTheta=0, rotationCenter=Vector(1, 1, -1))
        expectedPosition = Vector(2, 0, 0)
        self.assertAlmostEqual(expectedPosition.x, self.solid.position.x)
        self.assertAlmostEqual(expectedPosition.y, self.solid.position.y)
        self.assertAlmostEqual(expectedPosition.z, self.solid.position.z)

    def testWhenTranslate_shouldTranslateBBoxOfSolidAndPolygons(self):
        polygon = self.createPolygonMock()
        self.CUBOID_SURFACES.setPolygons('front', [polygon])
        solid = Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                      surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)
        oldBbox = solid.bbox

        solid.translateTo(Vector(1, -1, -1))

        # once during the __init__, once during the positioning, once during the translation = 3
        verify(polygon, times=3).resetBoundingBox()
        self.assertNotEqual(oldBbox, solid.bbox)

    def testShouldNotBeAStack(self):
        self.assertFalse(self.solid.isStack())

    def testGivenASolidWithInterfaces_shouldBeAStack(self):
        self.solid.surfaces.add(INTERFACE_KEY + "_1", [])
        self.assertTrue(self.solid.isStack())

    def testWhenSmooth_shouldSetVertexNormalAsAverageOfAdjacentPolygonNormals(self):
        self.solid.smooth()

        frontVertex = self.solid.vertices[0]
        self.assertAlmostEqual(1/math.sqrt(3), frontVertex.normal.x)
        self.assertAlmostEqual(1/math.sqrt(3), frontVertex.normal.y)
        self.assertAlmostEqual(1/math.sqrt(3), frontVertex.normal.z)

    def testWhenSmoothWithSurfaceLabel_shouldOnlySmoothPolygonsFromThisSurface(self):
        self.solid.smooth("front")

        frontVertex = self.solid.vertices[0]
        self.assertEqual(Vector(0, 0, 1), frontVertex.normal)
        backVertex = self.solid.vertices[5]
        self.assertIsNone(backVertex.normal)

    def testWhenSetLabel_shouldChangeLabel(self):
        newLabel = "newLabel"
        self.solid.setLabel(newLabel)
        self.assertEqual(newLabel, self.solid.getLabel())

    def _testGivenNoSurfaces_whenCreateSolidWithAnyPrimitive_shouldRaiseException(self, anyPrimitive):
        with self.assertRaises(NotImplementedError):
            Solid(position=self.position, material=self.material, vertices=self.CUBOID_VERTICES,
                  surfaces=None, primitive=anyPrimitive)

    def testGivenNoSurfaces_whenCreateSolidWithTrianglePrimitive_shouldRaiseException(self):
        self._testGivenNoSurfaces_whenCreateSolidWithAnyPrimitive_shouldRaiseException(primitives.TRIANGLE)

    def testGivenNoSurfaces_whenCreateSolidWithQuadPrimitive_shouldRaiseException(self):
        self._testGivenNoSurfaces_whenCreateSolidWithAnyPrimitive_shouldRaiseException(primitives.QUAD)

    def testGivenNoSurfaces_whenCreateSolidWithAnotherPrimitive_shouldRaiseException(self):
        self._testGivenNoSurfaces_whenCreateSolidWithAnyPrimitive_shouldRaiseException("anotherPrimitive")

    def testWhenCheckIfContainsAVertexOutsideBBox_shouldReturnFalse(self):
        self.assertFalse(self.solid.contains(Vertex(0, 0, 0)))

    def testWhenCheckIfContainsAVertexInsideInternalBBox_shouldReturnTrue(self):
        self.assertTrue(self.solid.contains(Vertex(2, 2, 0)))

    def testWhenCheckIfContainsAVertexPartiallyInside_shouldWarnAndReturnFalse(self):
        otherVertices = [Vertex(1, 1, -0.5), Vertex(3, 1, -0.5), Vertex(3, 3, -0.5), Vertex(1, 3, -0.5)]
        self.CUBOID_VERTICES.extend(otherVertices)
        self.CUBOID_SURFACES.add('other', [Quad(*otherVertices)])
        self.solid = Solid(material=self.material, vertices=self.CUBOID_VERTICES,
                           surfaces=self.CUBOID_SURFACES, primitive=primitives.TRIANGLE)

        with self.assertWarns(RuntimeWarning):
            self.assertFalse(self.solid.contains(Vertex(2, 2, -0.75)))

    @staticmethod
    def createPolygonMock() -> Polygon:
        polygon = mock(Polygon)
        when(polygon).resetNormal().thenReturn()
        when(polygon).resetBoundingBox().thenReturn()
        when(polygon).resetCentroid().thenReturn()
        when(polygon).setInsideEnvironment(...).thenReturn()
        return polygon
