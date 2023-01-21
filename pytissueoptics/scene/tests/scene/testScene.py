import unittest

from mockito import mock, verify, when

from pytissueoptics.scene import Cuboid
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector, BoundingBox, Environment
from pytissueoptics.scene.solids import Solid


class TestScene(unittest.TestCase):
    WORLD_MATERIAL = "worldMaterial"

    def setUp(self):
        self.scene = Scene(worldMaterial=self.WORLD_MATERIAL)

    def testWhenAddingASolid_shouldContainTheSolid(self):
        SOLID = self.makeSolidWith()
        self.scene.add(SOLID)

        self.assertEqual(1, len(self.scene.getSolids()))
        self.assertEqual(SOLID, self.scene.getSolids()[0])

    def testWhenAddingASolidAtAPosition_shouldPlaceTheSolidAtTheDesiredPosition(self):
        SOLID_POSITION = Vector(4, 0, 1)
        SOLID = self.makeSolidWith()
        when(SOLID).translateTo(...).thenReturn()

        self.scene.add(SOLID, position=SOLID_POSITION)

        verify(SOLID).translateTo(SOLID_POSITION)

    def testWhenAddingASolidAtNoSpecificPosition_shouldKeepTheSolidAtItsPredefinedPosition(self):
        SOLID = self.makeSolidWith()
        self.scene.add(SOLID)
        verify(SOLID, times=0).translateTo(...)

    def testWhenAddingASolidThatPartlyOverlapsWithAnotherOne_shouldNotAdd(self):
        OTHER_SOLID = self.makeSolidWith(BoundingBox([2, 5], [2, 5], [2, 5]), name="Other-solid")
        SOLID = self.makeSolidWith(BoundingBox([0, 3], [0, 3], [0, 3]))
        self.scene.add(OTHER_SOLID)

        with self.assertRaises(Exception):
            self.scene.add(SOLID)

    def testWhenAddingASolidInsideAnotherOne_shouldUpdateOutsideMaterialOfThisSolid(self):
        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True, name="Outside-solid")
        SOLID = self.makeSolidWith(BoundingBox([1, 3], [1, 3], [1, 3]))
        OUTSIDE_SOLID_ENV = Environment("A material")
        when(OUTSIDE_SOLID).getEnvironment().thenReturn(OUTSIDE_SOLID_ENV)
        self.scene.add(OUTSIDE_SOLID)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideEnvironment(OUTSIDE_SOLID_ENV)

    def testWhenAddingASolidOverAnotherOne_shouldUpdateOutsideMaterialOfTheOtherSolid(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), name="Inside-solid")
        self.scene.add(INSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True)
        SOLID_ENV = Environment("A material")
        when(SOLID).getEnvironment().thenReturn(SOLID_ENV)

        self.scene.add(SOLID)

        verify(INSIDE_SOLID).setOutsideEnvironment(SOLID_ENV)

    def testWhenAddingASolidInsideMultipleOtherSolids_shouldUpdateOutsideMaterialOfThisSolid(self):
        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), name="Outside-solid")
        OUTSIDE_SOLID_ENV = Environment("Outside solid material")
        when(OUTSIDE_SOLID).getEnvironment().thenReturn(OUTSIDE_SOLID_ENV)
        self.scene.add(OUTSIDE_SOLID)

        TOPMOST_OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]),
                                                   contains=True, name="Topmost-outside-solid")
        self.scene.add(TOPMOST_OUTSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]))
        when(OUTSIDE_SOLID).contains(...).thenReturn(True)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideEnvironment(OUTSIDE_SOLID_ENV)

    def testWhenAddingASolidOverMultipleOtherSolids_shouldUpdateOutsideMaterialOfTheTopMostSolid(self):
        TOPMOST_INSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True,
                                                  name="Topmost-inside-solid")
        self.scene.add(TOPMOST_INSIDE_SOLID)

        INSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), name="Inside-solid")
        self.scene.add(INSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True)
        SOLID_MATERIAL = "A material"
        when(SOLID).getEnvironment().thenReturn(SOLID_MATERIAL)

        self.scene.add(SOLID)

        verify(TOPMOST_INSIDE_SOLID).setOutsideEnvironment(SOLID_MATERIAL)

    def testWhenAddingASolidThatFitsInsideOneButAlsoContainsOne_shouldUpdateOutsideMaterialOfThisSolidAndTheOneInside(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]), name="Inside-solid")
        self.scene.add(INSIDE_SOLID)

        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True, name="Outside-solid")
        OUTSIDE_SOLID_ENV = Environment("Outside material")
        when(OUTSIDE_SOLID).getEnvironment().thenReturn(OUTSIDE_SOLID_ENV)
        self.scene.add(OUTSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]))
        SOLID_ENV = Environment("Inside material")
        when(SOLID).getEnvironment().thenReturn(SOLID_ENV)
        when(SOLID).contains(...).thenReturn(False).thenReturn(True)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideEnvironment(OUTSIDE_SOLID_ENV)
        verify(INSIDE_SOLID).setOutsideEnvironment(SOLID_ENV)

    def testWhenAddingASolidInsideASolidStack_shouldRaiseNotImplementedError(self):
        CUBOID_STACK = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), contains=True,
                                          isStack=True, name="Cuboid-stack")
        self.scene.add(CUBOID_STACK)

        INSIDE_SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]))

        with self.assertRaises(NotImplementedError):
            self.scene.add(INSIDE_SOLID)

    def testGivenASceneThatIgnoresIntersections_whenAddingASolidThatPartlyMergesWithAnotherOne_shouldAddTheSolid(self):
        scene = Scene(ignoreIntersections=True)
        OTHER_SOLID = self.makeSolidWith(BoundingBox([2, 5], [2, 5], [2, 5]), name="Other-solid")
        SOLID = self.makeSolidWith(BoundingBox([0, 3], [0, 3], [0, 3]))
        scene.add(OTHER_SOLID)

        scene.add(SOLID)

    def testWhenGetBoundingBox_shouldReturnABoundingBoxThatExtendsToAllItsSolids(self):
        scene = Scene()
        SOLID1 = self.makeSolidWith(BoundingBox([-5, -3], [-4, -2], [-3, -1]), name="Solid1")
        SOLID2 = self.makeSolidWith(BoundingBox([0, 3], [0, 2], [0, 1]), name="Solid2")
        scene.add(SOLID1)
        scene.add(SOLID2)

        bbox = scene.getBoundingBox()

        self.assertEqual([-5, 3], bbox.xLim)
        self.assertEqual([-4, 2], bbox.yLim)
        self.assertEqual([-3, 1], bbox.zLim)

    def testGivenNoSolids_whenGetBoundingBox_shouldReturnNone(self):
        scene = Scene()
        self.assertIsNone(scene.getBoundingBox())

    def testWhenAddingASolidWithExistingLabel_shouldWarnAndRenameByIncrementingLabel(self):
        SOLID_NAME = "Solid"
        solid1 = self.makeSolidWith(name=SOLID_NAME)
        solid2 = self.makeSolidWith(name=SOLID_NAME + "_2")
        solid3 = self.makeSolidWith(name=SOLID_NAME)
        scene = Scene([solid1, solid2], ignoreIntersections=True)

        with self.assertWarns(UserWarning):
            scene.add(solid3)

        verify(solid1, times=0).setLabel(...)
        verify(solid2, times=0).setLabel(...)
        verify(solid3).setLabel(SOLID_NAME + "_3")

    def testWhenResetOutsideMaterial_shouldSetOutsideMaterialOfAllTheSolidsThatAreNotContained(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), name="InsideSolid")
        self.scene.add(INSIDE_SOLID)
        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True, name="solid")
        self.scene.add(SOLID)

        self.scene.resetOutsideMaterial()

        verify(SOLID, times=1).setOutsideEnvironment(Environment(self.WORLD_MATERIAL))
        verify(INSIDE_SOLID, times=0).setOutsideEnvironment(Environment(self.WORLD_MATERIAL))

    def testWhenCreatingSceneFromSolids_shouldAutomaticallyResetOutsideMaterialOfAllSolidsThatAreNotContained(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), name="InsideSolid")
        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True, name="solid")

        Scene([SOLID, INSIDE_SOLID], worldMaterial=self.WORLD_MATERIAL)

        verify(SOLID, times=1).setOutsideEnvironment(Environment(self.WORLD_MATERIAL))
        verify(INSIDE_SOLID, times=0).setOutsideEnvironment(Environment(self.WORLD_MATERIAL))

    def testWhenGetEnvironmentWithPositionContainedInASolid_shouldReturnEnvironmentOfThisSolid(self):
        SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), contains=True)
        self.scene.add(SOLID)

        env = self.scene.getEnvironmentAt(Vector(2, 2, 2))

        self.assertEqual(SOLID.getEnvironment(), env)

    def testWhenGetEnvironmentWithPositionOutsideAllSolids_shouldReturnWorldEnvironment(self):
        SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]))
        self.scene.add(SOLID)

        env = self.scene.getEnvironmentAt(Vector(0, 0, 0))

        self.assertEqual(self.scene.getWorldEnvironment(), env)

    def testWhenGetEnvironmentWithPositionContainedInAStack_shouldReturnEnvironmentOfProperStackLayer(self):
        frontLayer = Cuboid(1, 1, 1, material="frontMaterial")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial")
        backLayer = Cuboid(1, 1, 1, material="backMaterial")
        stack = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        self.scene.add(stack, position=Vector(0, 0, 0))

        frontEnv = self.scene.getEnvironmentAt(Vector(0, 0, -1))
        middleEnv = self.scene.getEnvironmentAt(Vector(0, 0, 0))
        backEnv = self.scene.getEnvironmentAt(Vector(0, 0, 1))

        self.assertEqual(Environment("frontMaterial", frontLayer), frontEnv)
        self.assertEqual(Environment("middleMaterial", middleLayer), middleEnv)
        self.assertEqual(Environment("backMaterial", backLayer), backEnv)

    def testWhenGetSolidFromLabel_shouldReturnTheSolid(self):
        SOLID_LABEL = "Solid"
        solid = self.makeSolidWith(name=SOLID_LABEL)
        self.scene.add(solid)

        returnedSolid = self.scene.getSolid(SOLID_LABEL)

        self.assertEqual(solid, returnedSolid)

    def testWhenGetSolidFromLabelThatDoesNotExist_shouldReturnNone(self):
        SOLID_LABEL = "Solid"
        solid = self.makeSolidWith(name=SOLID_LABEL)
        self.scene.add(solid)

        returnedSolid = self.scene.getSolid("NonExistingLabel")

        self.assertIsNone(returnedSolid)

    def testWhenGetSolidFromLabelWithCapitalizationError_shouldReturnTheSolid(self):
        SOLID_LABEL_CAPITALIZED = "Solid"
        solid = self.makeSolidWith(name=SOLID_LABEL_CAPITALIZED)
        self.scene.add(solid)

        returnedSolid = self.scene.getSolid(SOLID_LABEL_CAPITALIZED.lower())

        self.assertEqual(solid, returnedSolid)

    def testWhenGetSolidFromInternalStackLayerLabel_shouldReturnTheWholeStack(self):
        stack = self.makeSolidWith(name="stack", isStack=True)
        internalLayers = ["Layer1", "Layer2", "Layer3"]
        when(stack).getLayerLabels().thenReturn(internalLayers)
        self.scene.add(stack)

        returnedSolid = self.scene.getSolid(internalLayers[1])

        self.assertEqual(stack, returnedSolid)

    def testWhenGetSolidLabels_shouldReturnAllLabels(self):
        SOLID_LABEL_1 = "A Solid"
        solid1 = self.makeSolidWith(name=SOLID_LABEL_1, bbox=BoundingBox([0, 1], [0, 1], [0, 1]))
        SOLID_LABEL_2 = "Another Solid"
        solid2 = self.makeSolidWith(name=SOLID_LABEL_2, bbox=BoundingBox([2, 3], [2, 3], [2, 3]))
        self.scene.add(solid1)
        self.scene.add(solid2)

        labels = self.scene.getSolidLabels()

        self.assertEqual([SOLID_LABEL_1, SOLID_LABEL_2], labels)

    def testGivenASceneWithAStack_whenGetSolidLabels_shouldOnlyIncludeInternalStackLayers(self):
        stack = self.makeSolidWith(name="stack", isStack=True)
        internalLayers = ["Layer1", "Layer2"]
        when(stack).getLayerLabels().thenReturn(internalLayers)
        self.scene.add(stack)

        labels = self.scene.getSolidLabels()

        self.assertEqual(internalLayers, labels)

    def testWhenGetSurfaceLabelsOfSolid_shouldReturnAllItsSurfaceLabels(self):
        SOLID_LABEL_1 = "Solid 1"
        SOLID_LABEL_2 = "Solid 2"
        solid1 = self.makeSolidWith(name=SOLID_LABEL_1, bbox=BoundingBox([0, 1], [0, 1], [0, 1]))
        solid2 = self.makeSolidWith(name=SOLID_LABEL_2, bbox=BoundingBox([2, 3], [2, 3], [2, 3]))
        solid1.surfaceLabels = ["Surface1", "Surface2"]
        solid2.surfaceLabels = ["Surface3", "Surface4"]
        self.scene.add(solid1)
        self.scene.add(solid2)

        labels = self.scene.getSurfaceLabels(SOLID_LABEL_1)

        self.assertEqual(solid1.surfaceLabels, labels)

    def testWhenGetSurfaceLabelsOfSolidThatDoesNotExist_shouldReturnNoLabels(self):
        solid = self.makeSolidWith()
        solid.surfaceLabels = ["Surface1", "Surface2"]
        self.scene.add(solid)

        labels = self.scene.getSurfaceLabels("NonExistingLabel")

        self.assertEqual([], labels)

    def testWhenGetSurfaceLabelsOfAStack_shouldReturnTheSurfaceLabelsForAllItsLayers(self):
        STACK_LABEL = "Stack"
        stack = self.makeSolidWith(name=STACK_LABEL, isStack=True)
        stack.surfaceLabels = ["Surface1", "Surface2", "Surface3"]
        self.scene.add(stack)

        labels = self.scene.getSurfaceLabels(STACK_LABEL)

        self.assertEqual(stack.surfaceLabels, labels)

    def testWhenGetSurfaceLabelsOfAStackLayer_shouldReturnTheSurfaceLabelsOfThisLayer(self):
        stack = self.makeSolidWith(isStack=True)
        internalLayers = ["Layer1", "Layer2"]
        surfacesOfLayer1 = ["Surface1", "Surface2"]
        when(stack).getLayerLabels().thenReturn(internalLayers)
        when(stack).getLayerSurfaceLabels(internalLayers[0]).thenReturn(surfacesOfLayer1)
        self.scene.add(stack)

        labels = self.scene.getSurfaceLabels(internalLayers[0])

        self.assertEqual(surfacesOfLayer1, labels)

    def testGivenNoContainedSolids_shouldHaveNoContainedLabels(self):
        labels = self.scene.getContainedSolidLabels("Solid")

        self.assertEqual([], labels)

    def testGivenSolidContainedInAnotherSolid_shouldHaveContainedSolidLabel(self):
        SOLID_LABEL = "Solid"
        CONTAINED_LABEL = "Contained"
        solid = self.makeSolidWith(name=SOLID_LABEL, contains=True, bbox=BoundingBox([0, 4], [0, 4], [0, 4]))
        contained = self.makeSolidWith(name=CONTAINED_LABEL, bbox=BoundingBox([1, 3], [1, 3], [1, 3]))
        self.scene.add(solid)
        self.scene.add(contained)

        labelsContainedBySolid = self.scene.getContainedSolidLabels(SOLID_LABEL)
        labelsContainedByContained = self.scene.getContainedSolidLabels(CONTAINED_LABEL)

        self.assertEqual([CONTAINED_LABEL], labelsContainedBySolid)
        self.assertEqual([], labelsContainedByContained)

    def testWhenGetMaterials_shouldReturnAllMaterialsPresentInTheScene(self):
        layerMaterials = ["Material1", "Material2"]
        frontLayer = Cuboid(1, 1, 1, material=layerMaterials[0])
        middleLayer = Cuboid(1, 1, 1, material=layerMaterials[1])
        stack = middleLayer.stack(frontLayer, 'front')
        self.scene.add(stack)

        materials = self.scene.getMaterials()

        self.assertEqual([self.WORLD_MATERIAL] + layerMaterials, materials)

    def testWhenGetPolygons_shouldReturnAllPolygonsPresentInTheScene(self):
        solid1 = Cuboid(1, 1, 1, label="Solid1")
        solid2 = Cuboid(1, 1, 1, label="Solid2")
        self.scene.add(solid1)
        self.scene.add(solid2, position=Vector(0, 0, 2))

        polygons = self.scene.getPolygons()

        self.assertEqual(solid1.getPolygons() + solid2.getPolygons(), polygons)

    def testGivenTwoIdenticalScenes_shouldHaveTheSameHash(self):
        worldMaterial = "WorldMaterial"
        cuboidA = Cuboid(1, 1, 1)
        cuboidB = Cuboid(1, 1, 1)
        sceneA = Scene([cuboidA], worldMaterial=worldMaterial)
        sceneB = Scene([cuboidB], worldMaterial=worldMaterial)

        self.assertEqual(hash(sceneA), hash(sceneB))

    def testGivenTwoScenesThatOnlyDifferInWorldMaterial_shouldHaveDifferentHash(self):
        cuboidA = Cuboid(1, 1, 1)
        cuboidB = Cuboid(1, 1, 1)
        sceneA = Scene([cuboidA], worldMaterial="WorldMaterialA")
        sceneB = Scene([cuboidB], worldMaterial="WorldMaterialB")

        self.assertNotEqual(hash(sceneA), hash(sceneB))

    def testGivenTwoScenesThatDifferInSolidPlacement_shouldHaveDifferentHash(self):
        worldMaterial = "WorldMaterial"
        cuboidA = Cuboid(1, 1, 1, position=Vector(0, 0, 0))
        cuboidB = Cuboid(1, 1, 1, position=Vector(1, 1, 1))
        sceneA = Scene([cuboidA], worldMaterial=worldMaterial)
        sceneB = Scene([cuboidB], worldMaterial=worldMaterial)

        self.assertNotEqual(hash(sceneA), hash(sceneB))

    @staticmethod
    def makeSolidWith(bbox: BoundingBox = None, contains=False, isStack=False, name="solid"):
        solid = mock(Solid)
        when(solid).getLabel().thenReturn(name)
        when(solid).setLabel(...).thenReturn()
        when(solid).getBoundingBox().thenReturn(bbox)
        when(solid).isStack().thenReturn(isStack)
        when(solid).getVertices().thenReturn([])
        when(solid).setOutsideEnvironment(...).thenReturn()
        when(solid).getEnvironment().thenReturn(Environment("A material"))
        when(solid).getVertices().thenReturn([])
        when(solid).contains(...).thenReturn(contains)
        return solid
