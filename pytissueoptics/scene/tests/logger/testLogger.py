import unittest

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger import Logger


class TestLogger(unittest.TestCase):
    def testGivenNewLogger_shouldBeEmpty(self):
        logger = Logger()

        self.assertEqual(0, len(logger.points))
        self.assertEqual(0, len(logger.dataPoints))
        self.assertEqual(0, len(logger.segments))

    def testWhenLogNewPoint_shouldAddPointToTheLoggedPoints(self):
        logger = Logger()
        logger.logPoint(Vector(0, 0, 0))
        logger.logPoint(Vector(1, 0, 0))
        newPoint = Vector(2, 0, 0)

        logger.logPoint(newPoint)

        self.assertEqual(3, len(logger.points))
        self.assertEqual(newPoint, logger.points[-1])

    def testWhenLogNewDataPoint_shouldAddDataPointToTheLoggedDataPoints(self):
        logger = Logger()
        logger.logDataPoint(-10.5, Vector(0, 0, 0))
        logger.logDataPoint(20.5, Vector(1, 0, 0))

        logger.logDataPoint(10, Vector(2, 0, 0))

        self.assertEqual(3, len(logger.dataPoints))
        self.assertEqual(10, logger.dataPoints[-1].value)
        self.assertEqual(Vector(2, 0, 0), logger.dataPoints[-1].position)

    def testWhenLogNewSegment_shouldAddSegmentToTheLoggedSegments(self):
        logger = Logger()
        logger.logSegment(Vector(0, 0, 0), Vector(0, 0, 1))
        logger.logSegment(Vector(0, 0, 0), Vector(0, 1, 0))

        logger.logSegment(Vector(0, 0, 0), Vector(1, 0, 0))

        self.assertEqual(3, len(logger.segments))
        self.assertEqual(Vector(0, 0, 0), logger.segments[-1].start)
        self.assertEqual(Vector(1, 0, 0), logger.segments[-1].end)

    def testGivenLoggerWithMultipleDataPoints_whenGetMinDataPoint_shouldReturnDataPointWithMinimumValue(self):
        logger = Logger()
        logger.logDataPoint(-10.5, Vector(0, 0, 0))
        logger.logDataPoint(10, Vector(2, 0, 0))
        logger.logDataPoint(20.5, Vector(1, 0, 0))

        minDataPoint = logger.getMinDataPoint()

        self.assertEqual(-10.5, minDataPoint.value)
        self.assertEqual(Vector(0, 0, 0), minDataPoint.position)

    def testGivenLoggerWithMultipleDataPoints_whenGetMaxDataPoint_shouldReturnDataPointWithMaximumValue(self):
        logger = Logger()
        logger.logDataPoint(-10.5, Vector(0, 0, 0))
        logger.logDataPoint(10, Vector(2, 0, 0))
        logger.logDataPoint(20.5, Vector(1, 0, 0))

        maxDataPoint = logger.getMaxDataPoint()

        self.assertEqual(20.5, maxDataPoint.value)
        self.assertEqual(Vector(1, 0, 0), maxDataPoint.position)
