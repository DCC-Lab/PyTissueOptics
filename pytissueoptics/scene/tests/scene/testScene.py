import unittest

from mockito import mock, verify, when

from pytissueoptics.scene import Cuboid
from pytissueoptics.scene.geometry import INTERFACE_KEY, BoundingBox, Environment, Vector
from pytissueoptics.scene.scene import Scene
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
        self.scene.add(OUTSIDE_SOLID)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideEnvironment(OUTSIDE_SOLID.getEnvironment())

    def testWhenAddingASolidOverAnotherOne_shouldUpdateOutsideMaterialOfTheOtherSolid(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), name="Inside-solid")
        self.scene.add(INSIDE_SOLID)
        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True)

        self.scene.add(SOLID)

        verify(INSIDE_SOLID).setOutsideEnvironment(SOLID.getEnvironment())

    def testWhenAddingASolidInsideMultipleOtherSolids_shouldUpdateOutsideMaterialOfThisSolid(self):
        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), name="Outside-solid")
        self.scene.add(OUTSIDE_SOLID)

        TOPMOST_OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]),
                                                   contains=True, name="Topmost-outside-solid")
        self.scene.add(TOPMOST_OUTSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]))
        when(OUTSIDE_SOLID).contains(...).thenReturn(True)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideEnvironment(OUTSIDE_SOLID.getEnvironment())

    def testWhenAddingASolidOverMultipleOtherSolids_shouldUpdateOutsideMaterialOfTheTopMostSolid(self):
        TOPMOST_INSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True,
                                                  name="Topmost-inside-solid")
        self.scene.add(TOPMOST_INSIDE_SOLID)

        INSIDE_SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]), name="Inside-solid")
        self.scene.add(INSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([-1, 6], [-1, 6], [-1, 6]), contains=True)
        self.scene.add(SOLID)

        verify(TOPMOST_INSIDE_SOLID).setOutsideEnvironment(SOLID.getEnvironment())

    def testWhenAddingASolidThatFitsInsideOneButAlsoContainsOne_shouldUpdateOutsideMaterialOfThisSolidAndTheOneInside(self):
        INSIDE_SOLID = self.makeSolidWith(BoundingBox([2, 3], [2, 3], [2, 3]), name="Inside-solid")
        self.scene.add(INSIDE_SOLID)

        OUTSIDE_SOLID = self.makeSolidWith(BoundingBox([0, 5], [0, 5], [0, 5]), contains=True, name="Outside-solid")
        self.scene.add(OUTSIDE_SOLID)

        SOLID = self.makeSolidWith(BoundingBox([1, 4], [1, 4], [1, 4]))
        when(SOLID).contains(...).thenReturn(False).thenReturn(True)

        self.scene.add(SOLID)

        verify(SOLID).setOutsideEnvironment(OUTSIDE_SOLID.getEnvironment())
        verify(INSIDE_SOLID).setOutsideEnvironment(SOLID.getEnvironment())

    def testWhenAddingASolidInsideASolidStack_shouldUpdateOutsideMaterialOfThisSolidToTheProperStackLayer(self):
        frontLayer = Cuboid(1, 1, 1, material="frontMaterial", label="frontLayer")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial", label="middleLayer")
        backLayer = Cuboid(1, 1, 1, material="backMaterial", label="backLayer")
        CUBOID_STACK = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        self.scene.add(CUBOID_STACK, position=Vector(0, 0, 0))

        SOLID_INSIDE_MIDDLE_LAYER = Cuboid(0.9, 0.9, 0.9, material="insideMaterial", label="insideSolid")

        self.scene.add(SOLID_INSIDE_MIDDLE_LAYER)

        anyPolygonOfSolidInside = SOLID_INSIDE_MIDDLE_LAYER.getPolygons()[0]
        self.assertEqual(anyPolygonOfSolidInside.outsideEnvironment, middleLayer.getEnvironment())

    def testWhenAddingASolidInsideASolidStackThatWasMovedAndRotated_shouldStillUpdateOutsideMaterialOfThisSolidToTheProperStackLayer(self):
        positionOffset = Vector(0, 10, 0)
        rotations = {'yTheta': 35}
        frontLayer = Cuboid(1, 1, 1, material="frontMaterial", label="frontLayer")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial", label="middleLayer")
        backLayer = Cuboid(1, 1, 1, material="backMaterial", label="backLayer")
        CUBOID_STACK = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        CUBOID_STACK.translateTo(positionOffset)
        CUBOID_STACK.rotate(**rotations)
        self.scene.add(CUBOID_STACK)

        SOLID_INSIDE_MIDDLE_LAYER = Cuboid(0.9, 0.9, 0.9, material="insideMaterial", label="insideSolid")
        SOLID_INSIDE_MIDDLE_LAYER.translateTo(positionOffset)
        SOLID_INSIDE_MIDDLE_LAYER.rotate(**rotations)

        self.scene.add(SOLID_INSIDE_MIDDLE_LAYER)

        anyPolygonOfSolidInside = SOLID_INSIDE_MIDDLE_LAYER.getPolygons()[0]
        self.assertEqual(anyPolygonOfSolidInside.outsideEnvironment, middleLayer.getEnvironment())

    def testWhenAddingAStackOverAnotherSolid_shouldUpdateOutsideMaterialOfTheOtherSolidWithTheProperStackLayer(self):
        SOLID_INSIDE_FRONT_LAYER = Cuboid(0.9, 0.9, 0.9, material="insideMaterial", label="insideSolid")
        self.scene.add(SOLID_INSIDE_FRONT_LAYER, position=Vector(0, 0, -1))

        frontLayer = Cuboid(1, 1, 1, material="frontMaterial", label="frontLayer")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial", label="middleLayer")
        backLayer = Cuboid(1, 1, 1, material="backMaterial", label="backLayer")
        CUBOID_STACK = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        self.scene.add(CUBOID_STACK, position=Vector(0, 0, 0))

        anyPolygonOfSolidInside = SOLID_INSIDE_FRONT_LAYER.getPolygons()[0]
        self.assertEqual(anyPolygonOfSolidInside.outsideEnvironment, frontLayer.getEnvironment())

    def testWhenAddingASolidThatPartiallyOverlapsWithInternalStackLayers_shouldNotAdd(self):
        frontLayer = Cuboid(1, 1, 1, material="frontMaterial", label="frontLayer")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial", label="middleLayer")
        backLayer = Cuboid(1, 1, 1, material="backMaterial", label="backLayer")
        CUBOID_STACK = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        self.scene.add(CUBOID_STACK, position=Vector(0, 0, 0))

        SOLID = Cuboid(0.9, 1.1, 0.9, material="outsideMaterial", label="outsideSolid")

        with self.assertRaises(Exception):
            self.scene.add(SOLID)

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
        frontLayer = Cuboid(1, 1, 1, material="frontMaterial", label="frontLayer")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial", label="middleLayer")
        backLayer = Cuboid(1, 1, 1, material="backMaterial", label="backLayer")
        stack = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        self.scene.add(stack, position=Vector(0, 0, 0))

        frontEnv = self.scene.getEnvironmentAt(Vector(0, 0, -1))
        middleEnv = self.scene.getEnvironmentAt(Vector(0, 0, 0))
        backEnv = self.scene.getEnvironmentAt(Vector(0, 0, 1))

        self.assertEqual(Environment("frontMaterial", frontLayer), frontEnv)
        self.assertEqual(Environment("middleMaterial", middleLayer), middleEnv)
        self.assertEqual(Environment("backMaterial", backLayer), backEnv)

    def testWhenGetEnvironmentWithPositionInsideAContainedSolid_shouldReturnEnvironmentOfThisContainedSolid(self):
        SOLID = Cuboid(3, 3, 3, material="Material of solid", label="Solid")
        CONTAINED_SOLID = Cuboid(2, 2, 2, material="Material of contained solid", label="Contained solid")

        self.scene.add(SOLID)
        self.scene.add(CONTAINED_SOLID)

        env = self.scene.getEnvironmentAt(Vector(0, 0, 0))

        self.assertEqual(CONTAINED_SOLID.getEnvironment(), env)

    def testWhenGetSolidFromLabel_shouldReturnTheSolid(self):
        SOLID_LABEL = "Solid"
        solid = self.makeSolidWith(name=SOLID_LABEL)
        self.scene.add(solid)

        returnedSolid = self.scene.getSolid(SOLID_LABEL)

        self.assertEqual(solid, returnedSolid)

    def testWhenGetSolidFromLabelThatDoesNotExist_shouldRaise(self):
        SOLID_LABEL = "Solid"
        solid = self.makeSolidWith(name=SOLID_LABEL)
        self.scene.add(solid)

        with self.assertRaises(ValueError):
            self.scene.getSolid("NonExistingLabel")

    def testWhenGetSolidFromLabelWithCapitalizationError_shouldReturnTheSolid(self):
        SOLID_LABEL_CAPITALIZED = "Solid"
        solid = self.makeSolidWith(name=SOLID_LABEL_CAPITALIZED)
        self.scene.add(solid)

        returnedSolid = self.scene.getSolid(SOLID_LABEL_CAPITALIZED.lower())

        self.assertEqual(solid, returnedSolid)

    def testWhenGetSolidFromInternalStackLayerLabel_shouldReturnTheWholeStack(self):
        frontLayer = Cuboid(1, 1, 1, material="frontMaterial", label="frontLayer")
        middleLayer = Cuboid(1, 1, 1, material="middleMaterial", label="middleLayer")
        backLayer = Cuboid(1, 1, 1, material="backMaterial", label="backLayer")
        stack = backLayer.stack(middleLayer, 'front').stack(frontLayer, 'front')
        self.scene.add(stack)

        returnedSolid = self.scene.getSolid("frontLayer")

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
        internalLayers = ["Layer1", "Layer2"]
        layer1 = Cuboid(1, 1, 1, label=internalLayers[0])
        layer2 = Cuboid(1, 1, 1, label=internalLayers[1])
        stack = layer1.stack(layer2, 'front')
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

    def testWhenGetSurfaceLabelsOfSolidThatDoesNotExist_shouldRaise(self):
        solid = self.makeSolidWith()
        solid.surfaceLabels = ["Surface1", "Surface2"]
        self.scene.add(solid)

        with self.assertRaises(ValueError):
            self.scene.getSurfaceLabels("NonExistingLabel")

    def testWhenGetSurfaceLabelsOfAStack_shouldReturnTheSurfaceLabelsForAllItsLayers(self):
        STACK_LABEL = "Stack"
        internalLayers = ["Layer1", "Layer2"]
        layer1 = Cuboid(1, 1, 1, label=internalLayers[0])
        layer2 = Cuboid(1, 1, 1, label=internalLayers[1])
        stack = layer1.stack(layer2, 'front', stackLabel=STACK_LABEL)
        self.scene.add(stack)

        labels = self.scene.getSurfaceLabels(STACK_LABEL)

        self.assertEqual(stack.surfaceLabels, labels)

    def testWhenGetSurfaceLabelsOfAStackLayer_shouldReturnTheSurfaceLabelsOfThisLayer(self):
        internalLayers = ["Layer1", "Layer2"]
        layer1 = Cuboid(1, 1, 1, label=internalLayers[0])
        layer2 = Cuboid(1, 1, 1, label=internalLayers[1])
        stack = layer1.stack(layer2, 'front')
        self.scene.add(stack)

        labels = self.scene.getSurfaceLabels(internalLayers[0])

        self.assertEqual(6, len(labels))
        self.assertTrue(INTERFACE_KEY+"0" in labels)

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
        frontLayer = Cuboid(1, 1, 1, material=layerMaterials[0], label="Front Layer")
        middleLayer = Cuboid(1, 1, 1, material=layerMaterials[1], label="Middle Layer")
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
    def makeSolidWith(bbox: BoundingBox = None, contains=False, name="solid") -> Solid:
        solid = mock(Solid)
        when(solid).getLabel().thenReturn(name)
        when(solid).setLabel(...).thenReturn()
        when(solid).getBoundingBox().thenReturn(bbox)
        when(solid).getVertices().thenReturn([])
        when(solid).isStack().thenReturn(False)
        when(solid).setOutsideEnvironment(...).thenReturn()
        when(solid).getEnvironment().thenReturn(Environment("A material", solid))
        when(solid).getVertices().thenReturn([])
        when(solid).contains(...).thenReturn(contains)
        if bbox:
            solid.position = bbox.center
        return solid
