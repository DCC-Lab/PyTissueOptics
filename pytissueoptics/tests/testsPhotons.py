import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")

class TestPhoton(envtest.PyTissueTestCase):

    def testPhoton(self):
        p = Photon()
        self.assertIsNotNone(p)
        self.assertEqual(p.r, Vector(0,0,0))
        self.assertEqual(p.ez, Vector(0,0,1))

    def testSurface(self):
        s=Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        self.assertEqual(s.normal, zHat)

    def testReflectNormalIncidence(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=zHat)

        thetaIn, planeOfIncidenceNormal, actualNormal = p.ez.angleOfIncidence(s.normal)
        p.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)
        self.assertAlmostEqual((p.ez - -zHat).norm(), 0, 6)

    def testReflectPlus45Incidence(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=Vector(0,1,1).normalized())

        thetaIn, planeOfIncidenceNormal, actualNormal = p.ez.angleOfIncidence(s.normal)
        p.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)
        self.assertAlmostEqual((p.ez - Vector(0,1,-1).normalized()).norm(), 0, 6)

    def testReflectMinus45Incidence(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=Vector(0,-1,1).normalized())

        thetaIn, planeOfIncidenceNormal, actualNormal = p.ez.angleOfIncidence(s.normal)
        p.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)
        self.assertAlmostEqual((p.ez - Vector(0,-1,-1).normalized()).norm(), 0, 6)

    def testReflect(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=zHat)
        p.reflect(s)
        self.assertTrue( (p.ez+zHat).norm() < 1e-6)

    def testRefractInto(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=-zHat)
        s.indexInside = 1.4
        s.indexOutside = 1.0

        p = Photon(position=Vector(0,0,0), direction=zHat)
        p.refract(s)
        self.assertTrue( (p.ez-zHat).norm() < 1e-6)


if __name__ == '__main__':
    envtest.main()
