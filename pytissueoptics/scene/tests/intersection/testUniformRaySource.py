import math
import unittest

from pytissueoptics.scene import Vector
from pytissueoptics.scene.intersection import UniformRaySource


class TestUniformRaySource(unittest.TestCase):
    def testShouldInitializeAllRayOriginsAtTheSourcePosition(self):
        sourcePosition = Vector(3, 3, 0)
        direction = Vector(0, 0, 1)
        xTheta, yTheta = 10, 20
        xRes, yRes = 16, 16

        source = UniformRaySource(sourcePosition, direction, xTheta, yTheta, xRes, yRes)

        for ray in source.rays:
            self.assertEqual(ray.origin, sourcePosition)

    def testShouldHaveAUniformAngularDistributionOfRaysDirectionCenteredAroundTheSourceDirection(self):
        sourcePosition = Vector(3, 3, 0)
        sourceXAngle = 30 * math.pi / 180
        sourceDirection = Vector(math.sin(sourceXAngle), 0, math.cos(sourceXAngle))
        xTheta, yTheta = 20, 10
        xRes, yRes = 5, 2

        source = UniformRaySource(sourcePosition, sourceDirection, xTheta, yTheta, xRes, yRes)

        expectedXAngles = [-10, -5, 0, 5, 10]
        expectedYAngles = [-5, 5]
        for i, ray in enumerate(source.rays):
            xAngle, yAngle = self._getXYAngleOfDirection(ray.direction)
            xAngle -= sourceXAngle * 180 / math.pi

            expectedYAngle = expectedYAngles[0] if i < 5 else expectedYAngles[1]
            self.assertAlmostEqual(expectedXAngles[i % 5], xAngle)
            self.assertAlmostEqual(expectedYAngle, yAngle)

    @staticmethod
    def _getXYAngleOfDirection(direction):
        return math.atan(direction.x / direction.z) * 180 / math.pi, math.asin(direction.y) * 180 / math.pi
