import filecmp
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from pytissueoptics.rayscattering.display.utils import Direction
from pytissueoptics.rayscattering.display.views import *


class TestView2D(unittest.TestCase):
    def testGivenViewWithNotOrthogonalDirections_shouldRaiseException(self):
        with self.assertRaises(AssertionError):
            View2D(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.Z_NEG)

    def testGivenViewWithoutSetContext_whenExtractData_shouldRaiseException(self):
        view = View2D(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.X_POS)
        with self.assertRaises(RuntimeError):
            view.extractData(np.array([[1, 0, 0, 0]]))

    def testGivenAbstractView_whenExtractData_shouldRaiseException(self):
        abstractView = View2D(Direction.Z_POS, Direction.X_POS)
        abstractView.setContext([(0, 2), (-2, 2), (0, 3)], (0.1, 0.2, 0.3))
        with self.assertRaises(NotImplementedError):
            abstractView.extractData(np.array([[1, 0, 0, 0]]))

    def testShouldDefineAxisAsAxisOfProjectionDirection(self):
        view = View2D(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.X_POS)
        self.assertEqual(2, view.axis)

    def testShouldDefineAxisUAsAxisOfHorizontalDirection(self):
        view = View2D(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.X_POS)
        self.assertEqual(0, view.axisU)

    def testShouldDefineAxisVAsAxisOfVerticalDirection(self):
        view = View2D(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.X_POS)
        self.assertEqual(1, view.axisV)

    def testShouldHaveNoLimits(self):
        view = View2DProjectionX()
        self.assertIsNone(view.limitsU)
        self.assertIsNone(view.limitsV)

    def testShouldHaveNoBins(self):
        view = View2DProjectionX()
        self.assertIsNone(view.binsU)
        self.assertIsNone(view.binsV)

    def testWhenSetContext_shouldSetLimits(self):
        view = View2DProjectionX()
        limits3D = [(0, 2), (-2, 2), (0, 3)]
        binSize3D = (0.1, 0.2, 0.3)

        view.setContext(limits3D, binSize3D)

        self.assertEqual(list(limits3D[view.axisU]), view.limitsU)
        self.assertEqual(list(limits3D[view.axisV]), view.limitsV)

    def testWhenSetContext_shouldSetBins(self):
        view = View2DProjectionX()
        limits3D = [(0, 2), (-2, 2), (0, 3)]
        binSize3D = (0.1, 0.2, 0.3)

        view.setContext(limits3D, binSize3D)

        expectedBinsU = int((view.limitsU[1] - view.limitsU[0]) / binSize3D[view.axisU])
        expectedBinsV = int((view.limitsV[1] - view.limitsV[0]) / binSize3D[view.axisV])
        self.assertEqual(expectedBinsU, view.binsU)
        self.assertEqual(expectedBinsV, view.binsV)

    def testGivenLimits_whenSetContext_shouldNotChangeLimits(self):
        view = View2DProjectionX(limits=((-5, 5), (0, 10)))
        limits3D = [(0, 1), (0, 1), (0, 1)]
        binSize3D = (0.1, 0.2, 0.3)

        view.setContext(limits3D, binSize3D)

        self.assertEqual([-5, 5], view.limitsU)
        self.assertEqual([0, 10], view.limitsV)

    def testGivenBinSize_whenSetContext_shouldNotChangeBins(self):
        initialBinSize = 0.05
        view = View2DProjectionX(binSize=initialBinSize)
        limits3D = [(0, 2), (-2, 2), (0, 3)]
        binSize3D = (0.1, 0.2, 0.3)

        view.setContext(limits3D, binSize3D)

        expectedBinsU = int((view.limitsU[1] - view.limitsU[0]) / initialBinSize)
        expectedBinsV = int((view.limitsV[1] - view.limitsV[0]) / initialBinSize)
        self.assertEqual(expectedBinsU, view.binsU)
        self.assertEqual(expectedBinsV, view.binsV)

    def testGivenProjectionView_whenExtractData_shouldProject3DPointsToThis2DView(self):
        view = View2DProjectionX()
        view.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        value = 0.5
        dataPoints = np.array([[value, 0, 2.05, 2.05],
                               [value, 5, 2.05, 2.05],
                               [value, 0, 2.5, 2.95]])

        view.extractData(dataPoints)

        self.assertEqual(value * 3, view.getSum())

    def testWhenExtractDataWithNoData_shouldIgnore(self):
        view = View2DProjectionX()
        view.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        view.extractData(np.array([]))

        self.assertEqual(0, view.getSum())

    def testWhenGetImageDataWithoutLogScale_shouldReturnImageOfRawData(self):
        view = View2DProjectionX()
        view.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        value = 0.5
        dataPoints = np.array([[value, 0, 2.05, 2.05],
                               [value, 5, 2.05, 2.05],
                               [value, 0, 2.5, 2.95]])
        view.extractData(dataPoints)

        image = view.getImageData(logScale=False)

        self.assertEqual((view.binsU, view.binsV), image.shape)
        self.assertEqual(value * 3, image.sum())
        self.assertEqual(value * 2, image[0, -1])
        self.assertEqual(value, image[-1, 4])

    def testWhenGetImageDataWithLogScale_shouldReturnImageOfLogData(self):
        view = View2DProjectionX()
        view.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        value = 0.5
        dataPoints = np.array([[value, 0, 2.05, 2.05],
                               [value, 5, 2.05, 2.05],
                               [value, 0, 2.5, 2.95]])
        view.extractData(dataPoints)

        image = view.getImageData(logScale=True)

        self.assertEqual(1, image[0, -1])
        self.assertTrue(image.sum() > 1 + value)

    def testWhenGetImageDataWithoutAutoFlip_shouldPreserveRawUVDirections(self):
        view = View2DProjection(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.Y_NEG)
        view.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        value = 0.5
        dataPoints = np.array([[value, 2.05, 2.05, 0],
                               [value, 2.05, 2.05, 5],
                               [value, 2.5, 2.95, 0]])
        view.extractData(dataPoints)

        image = view.getImageData(logScale=False, autoFlip=False)

        self.assertEqual(value * 2, image[0, -1])
        self.assertEqual(value, image[-1, 4])

    def testWhenGetImageData_shouldAutoFlipAxesToBePositive(self):
        view = View2DProjection(projectionDirection=Direction.Z_POS, horizontalDirection=Direction.Y_NEG)
        view.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        value = 0.5
        dataPoints = np.array([[value, 2.05, 2.05, 0],
                               [value, 2.05, 2.05, 5],
                               [value, 2.5, 2.95, 0]])
        view.extractData(dataPoints)

        image = view.getImageData(logScale=False)

        self.assertEqual(value * 2, image[-1, 0])
        self.assertEqual(value, image[0, -5])

    def testGivenADefaultView_whenGetImageWithDefaultAlignment_shouldReturnSameImage(self):
        for defaultView in [View2DProjectionX(), View2DProjectionY(), View2DProjectionZ()]:
            with self.subTest(axis=defaultView.axis):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(defaultView)

    def testGivenXViewWithCustomHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.Y_POS, Direction.Y_NEG, Direction.Z_NEG]:
            customView = View2DProjection(Direction.X_POS, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection.name):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenXViewWithReverseProjectionAndAnyHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.Y_POS, Direction.Y_NEG, Direction.Z_POS, Direction.Z_NEG]:
            customView = View2DProjection(Direction.X_NEG, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection.name):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenYViewWithCustomHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.X_NEG, Direction.Z_NEG]:
            customView = View2DProjection(Direction.Y_NEG, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection.name):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenYViewWithReverseProjectionAndAnyHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.X_NEG, Direction.Z_POS, Direction.Z_NEG]:
            customView = View2DProjection(Direction.Y_POS, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection.name):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenZViewWithCustomHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.Y_POS, Direction.Y_NEG]:
            customView = View2DProjection(Direction.Z_POS, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection.name):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenZViewWithReverseProjectionAndAnyHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.X_NEG, Direction.Y_POS, Direction.Y_NEG]:
            customView = View2DProjection(Direction.Z_NEG, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection.name):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def _shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self, customView: View2D):
        if customView.axis == 0:
            naturalView = View2DProjectionX()
        elif customView.axis == 1:
            naturalView = View2DProjectionY()
        else:
            naturalView = View2DProjectionZ()
        dataPoints = np.array([[0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.95, 2.05, 2.5]])
        naturalView.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        customView.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        naturalView.extractData(dataPoints)
        customView.extractData(dataPoints)

        naturalImage = naturalView.getImageData()
        image = customView.getImageDataWithDefaultAlignment()

        self.assertTrue(np.array_equal(naturalImage, image))

    def testGivenProjectionView_whenExtractData_shouldIgnoreDataOutOfLimits(self):
        view = View2DProjectionX()
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        value = 0.5
        dataPointOut = np.array([[value, 0, 0, 0]])
        dataPointIn = np.array([[value, 0, 2.5, 2.5]])

        view.extractData(dataPointOut)
        view.extractData(dataPointIn)

        self.assertEqual(value, view.getSum())

    def testGivenAProjectionViewEqual_shouldBeContainedByTheOther(self):
        solidLabel = "A"
        view1 = View2DProjectionX(solidLabel=solidLabel)
        view2 = View2DProjectionX(solidLabel=solidLabel)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertTrue(view1.isContainedBy(view2))

    def testGivenAViewWithDifferentBinSize_shouldNotBeContainedByTheOther(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionX()
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.2, 0.2, 0.2))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenAViewWithDifferentSolidLabel_shouldNotBeContainedByTheOther(self):
        view1 = View2DProjectionX(solidLabel="A label")
        view2 = View2DProjectionX(solidLabel="Another label")
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenAViewWithSameProjectionAxis_shouldBeContainedByTheOther(self):
        view1 = View2DProjection(Direction.X_POS, Direction.Y_POS)
        view2 = View2DProjection(Direction.X_NEG, Direction.Z_NEG)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertTrue(view1.isContainedBy(view2))

    def testGivenAViewWithDifferentProjectionAxis_shouldNotBeContainedByTheOther(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionY()
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenAViewBigger_shouldNotBeContainedByTheOther(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionX()
        view1.setContext([(2, 5), (2, 5), (2, 5)], (0.1, 0.1, 0.1))
        view2.setContext([(3, 4), (3, 4), (3, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    @unittest.skip("Not implemented yet")
    def testGivenAViewSmaller_shouldBeContainedByTheOther(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionX()
        view1.setContext([(3, 4), (3, 4), (3, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 5), (2, 5), (2, 5)], (0.1, 0.1, 0.1))
        self.assertTrue(view1.isContainedBy(view2))

    def testGivenAViewSmallerWithSameNumberOfBins_shouldNotBeContainedByTheOther(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionX()
        view1.setContext([(3, 4), (3, 4), (3, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.2, 0.2, 0.2))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenASurfaceViewEqual_shouldBeContainedByTheOther(self):
        view1 = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=True)
        view2 = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=True)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertTrue(view1.isContainedBy(view2))

    def testGivenASurfaceViewWithDifferentLabel_shouldNotBeContainedByTheOther(self):
        view1 = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=True)
        view2 = View2DSurfaceX(solidLabel="A", surfaceLabel="C", surfaceEnergyLeaving=True)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenASurfaceViewWithDifferentEnergyDirection_shouldNotBeContainedByTheOther(self):
        view1 = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=True)
        view2 = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=False)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenAViewThatIsNotASlice_shouldHaveNoThickness(self):
        view = View2DProjectionX()
        self.assertIsNone(view.thickness)

    def testGivenAViewThatIsNotASlice_shouldNotHaveDisplayPositionSet(self):
        view = View2DProjectionX()
        self.assertIsNone(view.displayPosition)

    def testGivenAView_shouldBeAbleToManuallySetDisplayPosition(self):
        view = View2DProjectionX()
        view.displayPosition = 3
        self.assertEqual(view.displayPosition, 3)

    def testGivenASliceView_shouldSetDefaultThicknessToBinSizeOfSliceAxis(self):
        view = View2DSliceX(position=3)
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.2, 0.3))
        self.assertEqual(view.thickness, 0.1)

    def testGivenASliceView_shouldSetDisplayPositionAsSlicePosition(self):
        view = View2DSliceX(position=3)
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.2, 0.3))
        self.assertEqual(view.displayPosition, 3)

    def testGivenASliceViewEqual_shouldBeContainedByTheOther(self):
        view1 = View2DSliceX(solidLabel="A", position=3, thickness=0.1)
        view2 = View2DSliceX(solidLabel="A", position=3, thickness=0.1)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertTrue(view1.isContainedBy(view2))

    def testGivenASliceViewWithDifferentPosition_shouldNotBeContainedByTheOther(self):
        view1 = View2DSliceX(solidLabel="A", position=3, thickness=0.1)
        view2 = View2DSliceX(solidLabel="A", position=4, thickness=0.1)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenASliceViewWithDifferentThickness_shouldNotBeContainedByTheOther(self):
        view1 = View2DSliceX(position=3, thickness=0.1)
        view2 = View2DSliceX(position=3, thickness=0.2)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isContainedBy(view2))

    def testGivenAViewNotContainedByTheOther_shouldNotBeEqual(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionY()
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isEqualTo(view2))

    def testGivenAViewWithOppositeProjectionDirection_shouldNotBeEqual(self):
        view1 = View2DProjection(Direction.X_POS, Direction.Z_POS)
        view2 = View2DProjection(Direction.X_NEG, Direction.Z_POS)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isEqualTo(view2))

    def testGivenAViewWithDifferentHorizontal_shouldNotBeEqual(self):
        view1 = View2DProjection(Direction.X_POS, Direction.Z_POS)
        view2 = View2DProjection(Direction.X_POS, Direction.Z_NEG)
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isEqualTo(view2))

    def testGivenAViewWithDifferentBins_shouldNotBeEqual(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionX()
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.2, 0.2, 0.2))
        self.assertFalse(view1.isEqualTo(view2))

    def testGivenAViewWithDifferentLimits_shouldNotBeEqual(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionX()
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.2, 0.2, 0.2))
        view2.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        self.assertFalse(view1.isEqualTo(view2))

    def testWhenFlip_shouldFlipViewToBeSeenFromBehind(self):
        defaultViews = [View2DProjectionX(), View2DProjectionY(), View2DProjectionZ()]
        viewsFromBehind = [View2DProjection(Direction.X_NEG, Direction.Z_NEG),
                           View2DProjection(Direction.Y_POS, Direction.Z_NEG),
                           View2DProjection(Direction.Z_NEG, Direction.X_POS)]
        for view, viewFromBehind in zip(defaultViews, viewsFromBehind):
            with self.subTest(axis=view.axis):
                self._whenFlip_shouldEqualOtherView(view, viewFromBehind)

    def _whenFlip_shouldEqualOtherView(self, view: View2D, otherView: View2D):
        dataPoints = np.array([[0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.95, 2.05, 2.5]])
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        otherView.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view.extractData(dataPoints)
        otherView.extractData(dataPoints)

        view.flip()

        self.assertTrue(view.isEqualTo(otherView))
        self.assertTrue(np.array_equal(view.getImageData(), otherView.getImageData()))

    def testGivenAViewWithNoSolidLabel_shouldBePartOfSceneViewGroup(self):
        view = View2DProjectionX()
        self.assertEqual(ViewGroup.SCENE, view.group)

    def testGivenAViewWithSolidLabel_shouldBePartOfSolidsViewGroup(self):
        view = View2DProjectionX(solidLabel="A")
        self.assertEqual(ViewGroup.SOLIDS, view.group)

    def testGivenASurfaceViewOfEnergyLeaving_shouldBePartOfSurfacesLeavingViewGroup(self):
        view = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=True)
        self.assertEqual(ViewGroup.SURFACES_LEAVING, view.group)

    def testGivenASurfaceViewOfEnergyEntering_shouldBePartOfSurfacesEnteringViewGroup(self):
        view = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=False)
        self.assertEqual(ViewGroup.SURFACES_ENTERING, view.group)

    def testWhenInitDataFromAViewThatDoesNotContainTheSameData_shouldRaiseError(self):
        view1 = View2DProjectionX()
        view2 = View2DProjectionY()
        view1.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        view2.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        with self.assertRaises(AssertionError):
            view1.initDataFrom(view2)

    def testWhenInitDataFromAnotherViewThatContainsThisView_shouldExtractTheCorrectData(self):
        viewGiver = View2DProjection(Direction.X_POS, Direction.Z_POS)
        for horizontalDirection in [Direction.Y_POS, Direction.Y_NEG, Direction.Z_NEG]:
            viewReceiver = View2DProjection(Direction.X_POS, horizontalDirection)
            with self.subTest(projection=Direction.X_POS.name, horizontal=horizontalDirection.name):
                self._whenInitDataFromAnotherView_shouldExtractTheCorrectData(viewGiver, viewReceiver)

    def testWhenInitDataFromAnotherViewThatContainsThisViewOpposite_shouldExtractTheCorrectData(self):
        viewGiver = View2DProjection(Direction.X_POS, Direction.Z_POS)
        for horizontalDirection in [Direction.Y_POS, Direction.Y_NEG, Direction.Z_POS, Direction.Z_NEG]:
            viewReceiver = View2DProjection(Direction.X_NEG, horizontalDirection)
            with self.subTest(projection=Direction.X_NEG.name, horizontal=horizontalDirection.name):
                self._whenInitDataFromAnotherView_shouldExtractTheCorrectData(viewGiver, viewReceiver)

    def _whenInitDataFromAnotherView_shouldExtractTheCorrectData(self, viewGiver: View2DProjection,
                                                                 viewReceiver: View2DProjection):
        viewExpected = View2DProjection(viewReceiver.projectionDirection, viewReceiver._horizontalDirection)
        viewGiver.setContext([(2, 3), (2, 3), (2, 3)], (0.2, 0.2, 0.2))
        viewReceiver.setContext([(2, 3), (2, 3), (2, 3)], (0.2, 0.2, 0.2))
        viewExpected.setContext([(2, 3), (2, 3), (2, 3)], (0.2, 0.2, 0.2))
        dataPoints = np.array([[0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.95, 2.05, 2.5]])
        viewGiver.extractData(dataPoints)
        viewExpected.extractData(dataPoints)

        viewReceiver.initDataFrom(viewGiver)

        self.assertTrue(np.array_equal(viewExpected.getImageData(), viewReceiver.getImageData()))

    def testShouldHaveProperName(self):
        view = View2DProjectionX()
        self.assertEqual("View2DProjectionX of Scene", view.name)
        view = View2DSurface(Direction.X_POS, Direction.Y_POS, "Solid", "Top", surfaceEnergyLeaving=False)
        self.assertEqual("View2DSurface of Solid surface Top (entering)", view.name)

    def testShouldHaveProperDescription(self):
        view = View2DProjection(Direction.X_POS, Direction.Y_POS)
        self.assertEqual(f"{view.name} towards X_POS with Y_POS horizontal.", view.description)

    def testGivenASurfaceViewOfEnergyLeaving_whenExtractSurfaceData_shouldOnlyStorePositiveEnergy(self):
        view = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=True)
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        dataPoints = np.array([[-0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.05, 2.05, 2.05],
                               [-0.5, 2.95, 2.05, 2.5]])
        view.extractData(dataPoints)

        self.assertEqual(0.5, view.getSum())

    def testGivenASurfaceViewOfEnergyEntering_whenExtractSurfaceData_shouldOnlyStoreNegativeEnergyAsPositive(self):
        view = View2DSurfaceX(solidLabel="A", surfaceLabel="B", surfaceEnergyLeaving=False)
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        dataPoints = np.array([[-0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.05, 2.05, 2.05],
                               [-0.5, 2.95, 2.05, 2.5]])
        view.extractData(dataPoints)

        self.assertEqual(1, view.getSum())

    def testGivenASliceView_whenExtractData_shouldOnlyProjectDataThatLiesInsideTheSlice(self):
        sliceViews = [View2DSliceX(position=2, thickness=0.1),
                      View2DSliceY(position=3, thickness=0.1),
                      View2DSliceZ(position=4, thickness=0.1)]
        for view in sliceViews:
            with self.subTest(axis=view.axis):
                view.setContext([(2, 5), (2, 5), (2, 5)], (0.1, 0.1, 0.1))
                dataPoints = np.array([[0.5, 2.00, 3.00, 4.00],
                                       [1.0, 2.04, 3.04, 4.04],
                                       [2.0, 2.10, 3.10, 4.10]])
                view.extractData(dataPoints)
                self.assertEqual(1.5, view.getSum())

    def testShouldHaveSizeInPhysicalUnits(self):
        view = View2DProjectionX()
        view.setContext([(2, 3), (2, 4), (2, 5)], (0.1, 0.1, 0.1))
        self.assertEqual((3, 2), view.size)

    # todo: visual test. maybe reroute plt.show to a temp file and compare it to a reference image.

    def testWhenShow_shouldPlotTheViewWithCorrectDataAndAxes(self):
        view = View2DProjection(Direction.X_POS, Direction.Y_POS)
        view.setContext([(2, 4), (2, 4), (2, 4)], (0.1, 0.1, 0.1))
        dataPoints = np.array([[0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.05, 2.05, 2.05],
                               [0.5, 2.95, 2.05, 2.5]])
        view.extractData(dataPoints)

        with patch("matplotlib.pyplot.show") as mockShow:
            view.show()
            mockShow.assert_called_once()

        TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'testImages')
        referenceImage = os.path.join(TEST_IMAGES_DIR, 'viewXPOS_uYPOS_vZNEG.png')

        OVERWRITE_REFERENCE_IMAGES = False
        if OVERWRITE_REFERENCE_IMAGES:
            plt.savefig(referenceImage)
            self.skipTest("Overwriting reference image")

        with tempfile.TemporaryDirectory() as tempdir:
            currentImage = os.path.join(tempdir, 'test.png')
            plt.savefig(currentImage)
            self.assertTrue(filecmp.cmp(referenceImage, currentImage))
