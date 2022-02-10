import unittest

from mockito import mock, verify, when

from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.geometry import Polygon
from pytissueoptics.scene.geometry import SurfaceCollection


class TestSurfaceCollection(unittest.TestCase):
    SURFACE_NAME = "Top"
    SURFACE_POLYGON = mock(Polygon)

    def setUp(self):
        self.surfaceCollection = SurfaceCollection()

        self.SURFACE_POLYGONS = [self.SURFACE_POLYGON]

    def testShouldBeEmpty(self):
        self.assertEqual(0, len(self.surfaceCollection.surfaceNames))
        self.assertEqual(0, len(self.surfaceCollection.getPolygons()))

    def testWhenAddSurface_shouldAddItsNameToTheCollection(self):
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        self.assertTrue(self.SURFACE_NAME in self.surfaceCollection.surfaceNames)

    def testWhenAddSurface_shouldAddItsPolygonsToTheCollection(self):
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        self.assertEqual(self.SURFACE_POLYGONS, self.surfaceCollection.getPolygons())

    def testWhenAddSurfaceWithExistingName_shouldNotAddToCollection(self):
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        with self.assertRaises(Exception):
            self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)

    def testWhenGetPolygonsOfNonexistentSurface_shouldNotReturn(self):
        with self.assertRaises(Exception):
            self.surfaceCollection.getPolygons("Nonexistent surface name")

    def testWhenGetPolygonsOfNoSpecificSurface_shouldReturnThePolygonsOfAllSurfaces(self):
        otherPolygons = [mock(Polygon)]
        allPolygons = self.SURFACE_POLYGONS + otherPolygons
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        self.surfaceCollection.add("Another surface name", otherPolygons)

        polygons = self.surfaceCollection.getPolygons()

        self.assertEqual(allPolygons, polygons)

    def testWhenSetPolygonsOfASurface_shouldReplaceTheSurfacePolygonsWithTheProvidedPolygons(self):
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        newPolygons = [mock(Polygon)]

        self.surfaceCollection.setPolygons(self.SURFACE_NAME, newPolygons)

        self.assertEqual(newPolygons, self.surfaceCollection.getPolygons(self.SURFACE_NAME))

    def testWhenSetPolygonsOfNonexistentSurface_shouldNotSetPolygons(self):
        with self.assertRaises(Exception):
            self.surfaceCollection.setPolygons("Nonexistent surface name", self.SURFACE_POLYGONS)

    def testWhenSetOutsideMaterialOfASurface_shouldSetItForAllItsPolygons(self):
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        when(self.SURFACE_POLYGON).setOutsideMaterial(...).thenReturn()
        newMaterial = Material()

        self.surfaceCollection.setOutsideMaterial(newMaterial, self.SURFACE_NAME)

        verify(self.SURFACE_POLYGON).setOutsideMaterial(newMaterial)

    def testWhenSetOutsideMaterialOfNonexistentSurface_shouldNotSetOutsideMaterial(self):
        newMaterial = Material()
        with self.assertRaises(Exception):
            self.surfaceCollection.setOutsideMaterial(newMaterial, "Nonexistent surface name")

    def testWhenGetInsideMaterialOfASurface_shouldReturnTheMaterialUnderThisSurface(self):
        MATERIAL_UNDER_SURFACE = Material()
        self.SURFACE_POLYGON.insideMaterial = MATERIAL_UNDER_SURFACE
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)

        self.assertEqual(MATERIAL_UNDER_SURFACE, self.surfaceCollection.getInsideMaterial(self.SURFACE_NAME))

    def testWhenGetInsideMaterialOfASurfaceComposedOfDifferentMaterials_shouldNotReturn(self):
        MATERIAL_A = Material()
        MATERIAL_B = Material()
        polygonWithMaterialA = mock(Polygon)
        polygonWithMaterialB = mock(Polygon)
        polygonWithMaterialA.insideMaterial = MATERIAL_A
        polygonWithMaterialB.insideMaterial = MATERIAL_B

        self.surfaceCollection.add(self.SURFACE_NAME, [polygonWithMaterialA, polygonWithMaterialB])

        with self.assertRaises(Exception):
            self.surfaceCollection.getInsideMaterial(self.SURFACE_NAME)

    def testWhenResetNormals_shouldResetNormalsOfAllPolygons(self):
        when(self.SURFACE_POLYGON).resetNormal().thenReturn()
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)

        self.surfaceCollection.resetNormals()

        verify(self.SURFACE_POLYGON).resetNormal()

    def testWhenExtendWithAnotherSurfaceCollection_shouldAddTheOtherSurfacesToItsCollection(self):
        otherSurfaceName = "Another surface name"
        otherSurfacePolygons = [mock(Polygon)]
        otherSurfaceCollection = SurfaceCollection()
        otherSurfaceCollection.add(otherSurfaceName, otherSurfacePolygons)
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)
        
        self.surfaceCollection.extend(otherSurfaceCollection)
        
        self.assertEqual(self.surfaceCollection.surfaceNames, [self.SURFACE_NAME, otherSurfaceName])
        self.assertEqual(self.surfaceCollection.getPolygons(), self.SURFACE_POLYGONS + otherSurfacePolygons)

    def testWhenExtendWithAnotherSurfaceCollectionContainingExistingSurfaces_shouldNotExtend(self):
        otherSurfacePolygons = [mock(Polygon)]
        otherSurfaceCollection = SurfaceCollection()
        otherSurfaceCollection.add(self.SURFACE_NAME, otherSurfacePolygons)
        self.surfaceCollection.add(self.SURFACE_NAME, self.SURFACE_POLYGONS)

        with self.assertRaises(Exception):
            self.surfaceCollection.extend(otherSurfaceCollection)
