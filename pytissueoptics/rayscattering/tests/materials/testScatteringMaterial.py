import math
import unittest
from unittest.mock import patch

from pytissueoptics.rayscattering.materials import ScatteringMaterial


class TestScatteringMaterial(unittest.TestCase):
    def testShouldHaveAlbedo(self):
        material = ScatteringMaterial(mu_s=8, mu_a=2, g=0.9, n=1.4)
        expectedAlbedo = 2 / (2 + 8)
        self.assertEqual(expectedAlbedo, material.getAlbedo())

    @patch('numpy.random.random')
    def testShouldHaveScatteringDistance(self, mockRandom):
        randomDistanceRatio = 0.5
        mockRandom.return_value = randomDistanceRatio
        material = ScatteringMaterial(mu_s=8, mu_a=2, g=0.9, n=1.4)

        expectedScatteringDistance = -math.log(randomDistanceRatio) / (2 + 8)
        self.assertEqual(expectedScatteringDistance, material.getScatteringDistance())

    def testShouldHaveScatteringAngles(self):
        material = ScatteringMaterial(mu_s=8, mu_a=2, g=1, n=1.4)
        angles = material.getScatteringAngles()
        self.assertEqual(2, len(angles))

    def testGivenFullAnisotropyFactor_shouldHaveZeroThetaScatteringAngle(self):
        material = ScatteringMaterial(mu_s=8, mu_a=2, g=1, n=1.4)
        theta, phi = material.getScatteringAngles()
        self.assertEqual(0, theta)

    @patch('numpy.random.random')
    def testShouldHaveThetaScatteringAngleBetween0AndPi(self, mockRandom):
        mockRandom.return_value = 0
        material = ScatteringMaterial(mu_s=8, mu_a=2, g=0, n=1.4)
        theta, _ = material.getScatteringAngles()
        self.assertEqual(math.pi, theta)

        mockRandom.return_value = 0.5
        theta, _ = material.getScatteringAngles()
        self.assertEqual(math.pi / 2, theta)

        mockRandom.return_value = 1
        theta, _ = material.getScatteringAngles()
        self.assertEqual(0, theta)

    @patch('numpy.random.random')
    def testShouldHavePhiScatteringAngleBetween0And2Pi(self, mockRandom):
        mockRandom.return_value = 0
        material = ScatteringMaterial(mu_s=8, mu_a=2, g=0, n=1.4)
        _, phi = material.getScatteringAngles()
        self.assertEqual(0, phi)

        mockRandom.return_value = 0.5
        _, phi = material.getScatteringAngles()
        self.assertEqual(math.pi, phi)

        mockRandom.return_value = 1
        _, phi = material.getScatteringAngles()
        self.assertEqual(2 * math.pi, phi)

    def testGivenVacuumMaterial_shouldHaveZeroAlbedo(self):
        vacuum = ScatteringMaterial()
        self.assertEqual(0, vacuum.getAlbedo())

    def testGivenVacuumMaterial_shouldHaveInfiniteScatteringDistance(self):
        vacuum = ScatteringMaterial()
        self.assertEqual(math.inf, vacuum.getScatteringDistance())

    def testGivenTwoMaterialsWithTheSameProperties_shouldHaveSameHash(self):
        material1 = ScatteringMaterial(mu_s=8, mu_a=2, g=0.9, n=1.4)
        material2 = ScatteringMaterial(mu_s=8, mu_a=2, g=0.9, n=1.4)
        self.assertEqual(hash(material1), hash(material2))

    def testGivenTwoMaterialsWithDifferentProperties_shouldHaveDifferentHash(self):
        material1 = ScatteringMaterial(mu_s=8, mu_a=2, g=0.9, n=1.4)
        material2 = ScatteringMaterial(mu_s=8, mu_a=2, g=0.9, n=1.5)
        self.assertNotEqual(hash(material1), hash(material2))
