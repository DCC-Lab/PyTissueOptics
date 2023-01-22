import unittest
from unittest.mock import patch
import tempfile
import filecmp
import os

import numpy as np

from pytissueoptics import Logger
from pytissueoptics.scene.solids import Cuboid, Sphere, Ellipsoid
from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.viewer.mayavi import MayaviViewer, ViewPointStyle

TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'testImages')

SHOW = False
OVERWRITE_TEST_IMAGES = False

_SAVE_ALONGSIDE_TEST_IMAGES = False


class TestMayaviViewer(unittest.TestCase):
    def setUp(self):
        self.viewer = MayaviViewer()

    def testWhenAddLogger_shouldDrawAllLoggerComponents(self):
        logger = self._getTestLogger()
        self.viewer.addLogger(logger)
        self._assertViewerDisplays("logger_natural")

    def testWhenAddScene_shouldDrawAllItsSolids(self):
        scene = self._getTestScene()
        self.viewer.addScene(scene)
        self._assertViewerDisplays("scene_natural")

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

    @patch('mayavi.mlab.show')
    def testWhenShow_shouldDisplayTheMayaviViewer(self, fakeShow):
        solid = Cuboid(1, 1, 1)
        self.viewer.add(solid)

        self.viewer.show()

        self.assertEqual(1, fakeShow.call_count)

    def _assertViewerDisplays(self, displayName: str):
        expectedImageFile = os.path.join(TEST_IMAGES_DIR, f"{displayName}.png")

        if OVERWRITE_TEST_IMAGES:
            self.viewer.save(expectedImageFile)
        elif _SAVE_ALONGSIDE_TEST_IMAGES:
            currentImageFile = os.path.join(TEST_IMAGES_DIR, f"{displayName}_current.png")
            self.viewer.save(currentImageFile)

        with tempfile.TemporaryDirectory() as tmpdir:
            currentImageFile = os.path.join(tmpdir, 'currentViewer.png')
            self.viewer.save(currentImageFile)
            imagesAreEqual = filecmp.cmp(currentImageFile, expectedImageFile)

        if SHOW:
            self.viewer.show()
        if OVERWRITE_TEST_IMAGES or _SAVE_ALONGSIDE_TEST_IMAGES:
            self.skipTest("Cannot test when saving test images.")
        self.assertTrue(imagesAreEqual)

    @staticmethod
    def _getTestLogger():
        logger = Logger()
        for i in range(3):
            logger.logPoint(Vector(0, i/3, 0))
            logger.logDataPoint(i/20, Vector(1, i/3, 0))
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
