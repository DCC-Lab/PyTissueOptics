import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pytissueoptics import Logger
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.solids import Cuboid, Ellipsoid, Sphere
from pytissueoptics.scene.tests import SHOW_VISUAL_TESTS, compareVisuals
from pytissueoptics.scene.viewer.mayavi import MayaviViewer, ViewPointStyle

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "testImages")

OVERWRITE_TEST_IMAGES = False


def patchMayaviShow(func):
    for module in ["show", "gcf", "figure", "clf", "triangular_mesh"]:
        func = patch("mayavi.mlab." + module)(func)
    return func


@unittest.skipIf(
    not SHOW_VISUAL_TESTS, "Visual tests are disabled. Set scene.tests.SHOW_VISUAL_TESTS to True to enable them."
)
class TestMayaviViewer(unittest.TestCase):
    def setUp(self):
        self.viewer = MayaviViewer()

    def testWhenAddLogger_shouldDrawAllLoggerComponents(self):
        logger = self._getTestLogger()
        self.viewer.addLogger(logger)
        self._assertViewerDisplays("logger_natural")

    def testGivenOpticsViewPoint_shouldDisplayFromOpticsViewPoint(self):
        self.viewer = MayaviViewer(viewPointStyle=ViewPointStyle.OPTICS)
        self.viewer.add(self._getSimpleSolid())
        self._assertViewerDisplays("solid_optics")

    def testGivenNaturalFrontViewPoint_shouldDisplayFromNaturalFrontViewPoint(self):
        self.viewer = MayaviViewer(viewPointStyle=ViewPointStyle.NATURAL_FRONT)
        self.viewer.add(self._getSimpleSolid())
        self._assertViewerDisplays("solid_natural_front")

    def testWhenAddSpecialTestSphere_shouldDrawCorrectly(self):
        solid = Sphere(1, order=0, position=Vector(0, 0, 10))
        self.viewer.add(solid, colorWithPosition=True, representation="surface", showNormals=True)
        self._assertViewerDisplays("sphere_normals")

    def testWhenAddImages_shouldDraw2DImagesCorrectly(self):
        self.viewer = MayaviViewer(viewPointStyle=ViewPointStyle.NATURAL)
        testImage = np.zeros((5, 5))
        testImage[4, 4] = 1
        for axis in range(3):
            self.viewer.addImage(testImage, size=(1, 1), minCorner=(-0.5, -0.5), axis=axis, position=0.5)
        self.viewer.addPoints(np.array([[0, 0, -1]]))

        self._assertViewerDisplays("images")

    def testWhenAddEmptyLogger_shouldNotRaise(self):
        emptyLogger = Logger()
        self.viewer.addLogger(emptyLogger)

    @patchMayaviShow
    def testWhenShow_shouldDisplayTheMayaviViewer(self, fakeShow, *args):
        solid = Cuboid(1, 1, 1)
        self.viewer.add(solid)

        self.viewer.show()

        self.assertEqual(1, fakeShow.call_count)

    def _assertViewerDisplays(self, displayName: str):
        expectedImageFile = os.path.join(TEST_IMAGES_DIR, f"{displayName}.png")

        if OVERWRITE_TEST_IMAGES:
            self.viewer.save(expectedImageFile)
            self.skipTest("Cannot test when saving test images.")

        with tempfile.TemporaryDirectory() as tmpdir:
            currentImageFile = os.path.join(tmpdir, "currentViewer.png")
            self.viewer.save(currentImageFile)
            self.viewer.close()

            isOK = compareVisuals(expectedImageFile, currentImageFile, title=f"TestMayaviViewer: {displayName}")

            if not isOK:
                self.fail("Visual test failed.")

    @staticmethod
    def _getTestLogger():
        logger = Logger()
        for i in range(3):
            logger.logPoint(Vector(0, i / 3, 0))
            logger.logDataPoint(i / 20, Vector(1, i / 3, 0), None)
        logger.logSegment(Vector(0, 1, 0), Vector(1, 0, 0))
        return logger

    @staticmethod
    def _getTestScene():
        cuboid = Cuboid(a=1, b=2, c=1, position=Vector(1, 0, 0))
        sphere = Sphere(radius=0.5, order=0, position=Vector(0, 0, 0))
        ellipsoid = Ellipsoid(a=1.3, b=1, c=1, order=0, position=Vector(-2, 0, 0))
        return Scene([cuboid, sphere, ellipsoid], ignoreIntersections=True)

    @staticmethod
    def _getSimpleSolid():
        return Cuboid(a=1.5, b=2, c=1)
