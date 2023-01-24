import unittest

import numpy as np

from pytissueoptics import Direction, View2DProjection, View2DProjectionY, View2DProjectionZ
from pytissueoptics.rayscattering.display.views import View2DProjectionX, View2D


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

    def testGivenADefaultView_whenGetImageWithDefaultAlignment_shouldReturnSameImage(self):
        dataPoints = np.array([[0.5, 0, 2.05, 2.05],
                               [0.5, 5, 2.05, 2.05],
                               [0.5, 0, 2.5, 2.95]])
        naturalView = View2DProjectionX()
        naturalView.setContext([(2, 3), (2, 3), (2, 3)], (0.1, 0.1, 0.1))
        naturalView.extractData(dataPoints)

        naturalImage = naturalView.getImageData()
        alignedImage = naturalView.getImageDataWithDefaultAlignment()

        self.assertTrue(np.array_equal(naturalImage, alignedImage))

    def testGivenXViewWithCustomHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.Y_POS, Direction.Y_NEG, Direction.Z_NEG]:
            customView = View2DProjection(Direction.X_POS, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenXViewWithReverseProjectionAndAnyHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.Y_POS, Direction.Y_NEG, Direction.Z_POS, Direction.Z_NEG]:
            customView = View2DProjection(Direction.X_NEG, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenYViewWithCustomHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.X_NEG, Direction.Z_NEG]:
            customView = View2DProjection(Direction.Y_NEG, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenYViewWithReverseProjectionAndAnyHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.X_NEG, Direction.Z_POS, Direction.Z_NEG]:
            customView = View2DProjection(Direction.Y_POS, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenZViewWithCustomHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.Y_POS, Direction.Y_NEG]:
            customView = View2DProjection(Direction.Z_POS, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection):
                self._shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(customView)

    def testGivenZViewWithReverseProjectionAndAnyHorizontal_shouldHaveImageDataWithDefaultAlignmentEqualToNaturalView(self):
        for horizontalDirection in [Direction.X_POS, Direction.X_NEG, Direction.Y_POS, Direction.Y_NEG]:
            customView = View2DProjection(Direction.Z_NEG, horizontalDirection=horizontalDirection)
            with self.subTest(horizontalDirection=horizontalDirection):
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
