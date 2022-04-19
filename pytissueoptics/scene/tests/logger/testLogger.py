import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger import Logger, InteractionKey


class TestLogger(unittest.TestCase):
    SOLID_NAME = "mySolid"
    SURFACE_NAME = "front"
    INTERACTION_KEY = InteractionKey(SOLID_NAME, SURFACE_NAME)

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
        sameKey = InteractionKey(self.SOLID_NAME, self.SURFACE_NAME)
        anotherKey = InteractionKey(self.SOLID_NAME, "another surface")
        anotherKeyWithoutSurface = InteractionKey(self.SOLID_NAME)

        logger = Logger()
        logger.logPoint(Vector(0, 0, 0), self.INTERACTION_KEY)
        logger.logPoint(Vector(1, 0, 0), sameKey)
        logger.logPoint(Vector(2, 0, 0), anotherKey)
        logger.logPoint(Vector(3, 0, 0), anotherKeyWithoutSurface)

        self.assertEqual(2, len(logger.getPoints(self.INTERACTION_KEY)))
