import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random
from math import cos

inf = float("+inf")

class TestVector(envtest.PyTissueTestCase):
    def testMateiralInit(self):
        mat = Material()
        self.assertIsNotNone(mat)

    def testMaterialG0(self):
        mat = Material(g=0)
        sumTheta = 0
        sumPhi = 0
        N = 1000000
        for i in range(N):
            theta, phi = mat.getScatteringAngles(None)
            sumTheta += cos(theta)
            sumPhi += phi
        self.assertAlmostEqual((sumTheta)/N, 0, 2)
        self.assertAlmostEqual(sumPhi/N, 3.1416, 2)

    def testMaterialG08(self):
        mat = Material(g=0.8)
        sumTheta = 0
        sumPhi = 0
        N = 1000000
        for i in range(N):
            theta, phi = mat.getScatteringAngles(None)
            sumTheta += cos(theta)
            sumPhi += phi
        self.assertAlmostEqual((sumTheta)/N, 0.8, 2)
        self.assertAlmostEqual(sumPhi/N, 3.1416, 2)

    def testMaterialG1(self):
        mat = Material(g=1)
        sumTheta = 0
        sumPhi = 0
        N = 1000000
        for i in range(N):
            theta, phi = mat.getScatteringAngles(None)
            sumTheta += cos(theta)
            sumPhi += phi
        self.assertAlmostEqual((sumTheta)/N, 1, 2)
        self.assertAlmostEqual(sumPhi/N, 3.1416, 2)

    def testMaterialG1(self):
        mat = Material(g=-0.5)
        sumTheta = 0
        sumPhi = 0
        N = 1000000
        for i in range(N):
            theta, phi = mat.getScatteringAngles(None)
            sumTheta += cos(theta)
            sumPhi += phi
        self.assertAlmostEqual((sumTheta)/N, -0.5, 2)
        self.assertAlmostEqual(sumPhi/N, 3.1416, 2)

    def testMaterialMus(self):
        mat = Material(mu_s = 10, mu_a = 0, g = 0.8)
        sumDist = 0
        N = 1000000
        for i in range(N):
            d = mat.getScatteringDistance(None)
            sumDist += d
        self.assertAlmostEqual((sumDist)/N, 1/mat.mu_s, 2)

    def testMaterialMua(self):
        mat = Material(mu_s = 0, mu_a = 20, g = 0.8)
        sumDist = 0
        N = 1000000
        for i in range(N):
            d = mat.getScatteringDistance(None)
            sumDist += d
        self.assertAlmostEqual((sumDist)/N, 1/mat.mu_a, 2)

    def testMaterialMut(self):
        mat = Material(mu_s = 10, mu_a = 20, g = 0.8)
        sumDist = 0
        N = 1000000
        for i in range(N):
            d = mat.getScatteringDistance(None)
            sumDist += d
        self.assertAlmostEqual((sumDist)/N, 1/mat.mu_t, 2)

    def testMaterialMut(self):
        mat = Material(mu_s = 0, mu_a = 0, g = 0.8)
        sumDist = 0
        N = 1000000
        for i in range(N):
            d = mat.getScatteringDistance(None)
            sumDist += d
        self.assertAlmostEqual((sumDist)/N, mat.veryFar, 2)

if __name__ == '__main__':
    envtest.main()
