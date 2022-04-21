import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger import Logger, InteractionKey


class TestLogger(unittest.TestCase):
    SOLID_LABEL = "mySolid"
    SURFACE_LABEL = "front"
    INTERACTION_KEY = InteractionKey(SOLID_LABEL, SURFACE_LABEL)

    def testGivenNewLogger_shouldBeEmpty(self):
        logger = Logger()

        self.assertEqual(0, len(logger.getPoints()))
        self.assertEqual(0, len(logger.getDataPoints()))
        self.assertEqual(0, len(logger.getSegments()))

    def testWhenLogNewPoint_shouldAddPointToTheLoggedPoints(self):
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(1, 0, 0), self.INTERACTION_KEY)
        newPoint = Vector(2, 0, 0)

        logger.logPoint(newPoint, self.INTERACTION_KEY)

        self.assertEqual(3, len(logger.getPoints()))
        self.assertEqual(newPoint, logger.getPoints()[-1])

    def testWhenLogNewDataPoint_shouldAddDataPointToTheLoggedDataPoints(self):
        logger = Logger()
        logger.logDataPoint(-10.5, Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logDataPoint(20.5, Vector(1, 0, 0), self.INTERACTION_KEY)

        logger.logDataPoint(10, Vector(2, 0, 0), self.INTERACTION_KEY)

        self.assertEqual(3, len(logger.getDataPoints()))
        self.assertEqual(10, logger.getDataPoints()[-1].value)
        self.assertEqual(Vector(2, 0, 0), logger.getDataPoints()[-1].position)

    def testWhenLogNewSegment_shouldAddSegmentToTheLoggedSegments(self):
        logger = Logger()
        logger.logSegment(Vector(0, 0, 0), Vector(0, 0, 1), self.INTERACTION_KEY)
        logger.logSegment(Vector(0, 0, 0), Vector(0, 1, 0), self.INTERACTION_KEY)

        logger.logSegment(Vector(0, 0, 0), Vector(1, 0, 0), self.INTERACTION_KEY)

        self.assertEqual(3, len(logger.getSegments()))
        self.assertEqual(Vector(0, 0, 0), logger.getSegments()[-1].start)
        self.assertEqual(Vector(1, 0, 0), logger.getSegments()[-1].end)

    def testWhenGetDataWithKey_shouldReturnDataStoredForThisKey(self):
        sameKey = InteractionKey(self.SOLID_LABEL, self.SURFACE_LABEL)
        anotherKey = InteractionKey(self.SOLID_LABEL, "another surface")
        anotherKeyWithoutSurface = InteractionKey(self.SOLID_LABEL)

        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(1, 0, 0), sameKey)
        logger.logPoint(Vector(2, 0, 0), anotherKey)
        logger.logPoint(Vector(3, 0, 0), anotherKeyWithoutSurface)

        self.assertEqual(2, len(logger.getPoints(self.INTERACTION_KEY)))
        self.assertEqual(1, len(logger.getPoints(anotherKeyWithoutSurface)))

    def testWhenGetSolidLabels_shouldReturnAListOfUniqueSolidLabels(self):
        solidLabel1 = "A label"
        solidLabel2 = "Another label"
        interactionKey1 = InteractionKey(solidLabel1, "front")
        interactionKey2 = InteractionKey(solidLabel2)
        interactionKey3 = InteractionKey(solidLabel2, "back")

        logger = Logger()
        aPoint = Vector(0, 0, 0)
        logger.logPoint(aPoint, interactionKey1)
        logger.logPoint(aPoint, interactionKey2)
        logger.logPoint(aPoint, interactionKey3)

        solidLabels = logger.getSolidLabels()

        self.assertEqual(2, len(solidLabels))
        self.assertTrue(solidLabel1 in solidLabels)
        self.assertTrue(solidLabel2 in solidLabels)

    def testWhenGetSurfaceLabels_shouldReturnAListOfUniqueSurfaceLabelsForTheDesiredSolid(self):
        solidLabel1 = "A label"
        solidLabel2 = "Another label"
        surfaceA = "front"
        surfaceB = "back"
        anotherSurface = "top"
        interactionKey1 = InteractionKey(solidLabel1, surfaceA)
        interactionKey2 = InteractionKey(solidLabel1, surfaceB)
        interactionKey3 = InteractionKey(solidLabel1)
        interactionKey4 = InteractionKey(solidLabel2, anotherSurface)

        logger = Logger()
        aPoint = Vector(0, 0, 0)
        logger.logPoint(aPoint, interactionKey1)
        logger.logPoint(aPoint, interactionKey2)
        logger.logPoint(aPoint, interactionKey3)
        logger.logPoint(aPoint, interactionKey4)

        surfaceLabels = logger.getSurfaceLabels(solidLabel1)

        self.assertEqual(2, len(surfaceLabels))
        self.assertTrue(surfaceA in surfaceLabels)
        self.assertTrue(surfaceB in surfaceLabels)
