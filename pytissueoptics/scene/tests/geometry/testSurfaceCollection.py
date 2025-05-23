import unittest

from mockito import mock, verify, when

from pytissueoptics.scene.geometry import INTERFACE_KEY, Environment, Polygon, SurfaceCollection


class TestSurfaceCollection(unittest.TestCase):
    SURFACE_LABEL = "top"
    SURFACE_POLYGON = mock(Polygon)

    def setUp(self):
        self.surfaceCollection = SurfaceCollection()

        self.SURFACE_POLYGONS = [self.SURFACE_POLYGON]

    def testShouldBeEmpty(self):
        self.assertEqual(0, len(self.surfaceCollection.surfaceLabels))
        self.assertEqual(0, len(self.surfaceCollection.getPolygons()))

    def testWhenAddSurface_shouldAddItsLabelToTheCollection(self):
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        self.assertTrue(self.SURFACE_LABEL in self.surfaceCollection.surfaceLabels)

    def testWhenAddSurface_shouldAddItsPolygonsToTheCollection(self):
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        self.assertEqual(self.SURFACE_POLYGONS, self.surfaceCollection.getPolygons())

    def testWhenGetPolygonsOfNonexistentSurface_shouldNotReturn(self):
        with self.assertRaises(Exception):
            self.surfaceCollection.getPolygons("Nonexistent surface label")

    def testWhenGetPolygonsOfNoSpecificSurface_shouldReturnThePolygonsOfAllSurfaces(self):
        otherPolygons = [mock(Polygon)]
        allPolygons = self.SURFACE_POLYGONS + otherPolygons
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        self.surfaceCollection.add("Another surface label", otherPolygons)

        polygons = self.surfaceCollection.getPolygons()

        self.assertEqual(allPolygons, polygons)

    def testWhenSetPolygonsOfASurface_shouldReplaceTheSurfacePolygonsWithTheProvidedPolygons(self):
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        newPolygons = [mock(Polygon)]

        self.surfaceCollection.setPolygons(self.SURFACE_LABEL, newPolygons)

        self.assertEqual(newPolygons, self.surfaceCollection.getPolygons(self.SURFACE_LABEL))

    def testWhenSetPolygonsOfNonexistentSurface_shouldNotSetPolygons(self):
        with self.assertRaises(Exception):
            self.surfaceCollection.setPolygons("Nonexistent surface label", self.SURFACE_POLYGONS)

    def testWhenSetOutsideEnvironment_shouldSetItForAllSurfacePolygons(self):
        polygonInSurfaceA = mock(Polygon)
        polygonInSurfaceB = mock(Polygon)
        when(polygonInSurfaceA).setOutsideEnvironment(...).thenReturn()
        when(polygonInSurfaceB).setOutsideEnvironment(...).thenReturn()
        self.surfaceCollection.add("Surface A", [polygonInSurfaceA])
        self.surfaceCollection.add("Surface B", [polygonInSurfaceB])
        newEnvironment = Environment("New material")

        self.surfaceCollection.setOutsideEnvironment(newEnvironment)

        verify(polygonInSurfaceA).setOutsideEnvironment(newEnvironment)
        verify(polygonInSurfaceB).setOutsideEnvironment(newEnvironment)

    def testGivenInterfaces_whenSetOutsideEnvironment_shouldNotSetOutsideEnvironmentOfInterfaces(self):
        polygonInInterface = mock(Polygon)
        polygonInSurface = mock(Polygon)
        when(polygonInInterface).setOutsideEnvironment(...).thenReturn()
        when(polygonInSurface).setOutsideEnvironment(...).thenReturn()
        self.surfaceCollection.add(f"A {INTERFACE_KEY}", [polygonInInterface])
        self.surfaceCollection.add("Surface", [polygonInSurface])
        newEnvironment = Environment("New material")

        self.surfaceCollection.setOutsideEnvironment(newEnvironment)

        verify(polygonInInterface, times=0).setOutsideEnvironment(...)
        verify(polygonInSurface).setOutsideEnvironment(newEnvironment)

    def testWhenSetOutsideEnvironmentOfASurface_shouldSetItForAllItsPolygons(self):
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        when(self.SURFACE_POLYGON).setOutsideEnvironment(...).thenReturn()
        newEnvironment = Environment("New material")

        self.surfaceCollection.setOutsideEnvironment(newEnvironment, self.SURFACE_LABEL)

        verify(self.SURFACE_POLYGON).setOutsideEnvironment(newEnvironment)

    def testWhenSetOutsideEnvironmentOfNonexistentSurface_shouldNotSetOutsideEnvironment(self):
        newEnvironment = Environment("New material")
        with self.assertRaises(Exception):
            self.surfaceCollection.setOutsideEnvironment(newEnvironment, "Nonexistent surface label")

    def testWhenGetInsideEnvironmentOfASurface_shouldReturnTheEnvironmentUnderThisSurface(self):
        ENVIRONMENT_UNDER_SURFACE = Environment("A Material")
        self.SURFACE_POLYGON.insideEnvironment = ENVIRONMENT_UNDER_SURFACE
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        self.assertEqual(ENVIRONMENT_UNDER_SURFACE, self.surfaceCollection.getInsideEnvironment(self.SURFACE_LABEL))

    def testWhenGetInsideEnvironmentOfASurfaceComposedOfDifferentEnvironments_shouldNotReturn(self):
        ENVIRONMENT_A = Environment("Material A")
        ENVIRONMENT_B = Environment("Material B")
        polygonWithEnvironmentA = mock(Polygon)
        polygonWithEnvironmentB = mock(Polygon)
        polygonWithEnvironmentA.insideEnvironment = ENVIRONMENT_A
        polygonWithEnvironmentB.insideEnvironment = ENVIRONMENT_B

        self.surfaceCollection.add(self.SURFACE_LABEL, [polygonWithEnvironmentA, polygonWithEnvironmentB])

        with self.assertRaises(Exception):
            self.surfaceCollection.getInsideEnvironment(self.SURFACE_LABEL)

    def testWhenResetNormals_shouldResetNormalOfAllPolygons(self):
        when(self.SURFACE_POLYGON).resetNormal().thenReturn()
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        self.surfaceCollection.resetNormals()

        verify(self.SURFACE_POLYGON).resetNormal()

    def testWhenResetBoundingBoxes_shouldResetBoundingBoxOfAllPolygons(self):
        when(self.SURFACE_POLYGON).resetBoundingBox().thenReturn()
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        self.surfaceCollection.resetBoundingBoxes()

        verify(self.SURFACE_POLYGON).resetBoundingBox()

    def testWhenResetCentroids_shouldResetCentroidOfAllPolygons(self):
        when(self.SURFACE_POLYGON).resetCentroid().thenReturn()
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        self.surfaceCollection.resetCentroids()

        verify(self.SURFACE_POLYGON).resetCentroid()

    def testWhenExtendWithAnotherSurfaceCollection_shouldAddTheOtherSurfacesToItsCollection(self):
        otherSurfaceLabel = "Another surface label"
        otherSurfacePolygons = [mock(Polygon)]
        otherSurfaceCollection = SurfaceCollection()
        otherSurfaceCollection.add(otherSurfaceLabel, otherSurfacePolygons)
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        self.surfaceCollection.extend(otherSurfaceCollection)

        self.assertEqual(self.surfaceCollection.surfaceLabels, [self.SURFACE_LABEL, otherSurfaceLabel])
        self.assertEqual(self.surfaceCollection.getPolygons(), self.SURFACE_POLYGONS + otherSurfacePolygons)

    def testWhenExtendWithAnotherSurfaceCollectionContainingExistingSurfaces_shouldNotExtend(self):
        otherSurfacePolygons = [mock(Polygon)]
        otherSurfaceCollection = SurfaceCollection()
        otherSurfaceCollection.add(self.SURFACE_LABEL, otherSurfacePolygons)
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        with self.assertRaises(Exception):
            self.surfaceCollection.extend(otherSurfaceCollection)

    def testWhenAddingASurfaceWithAnExistingLabel_shouldIncrementTheLabel(self):
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)
        self.surfaceCollection.add(self.SURFACE_LABEL, self.SURFACE_POLYGONS)

        newSurfaceLabels = [self.SURFACE_LABEL, f"{self.SURFACE_LABEL}_2", f"{self.SURFACE_LABEL}_3"]
        self.assertEqual(newSurfaceLabels, self.surfaceCollection.surfaceLabels)
