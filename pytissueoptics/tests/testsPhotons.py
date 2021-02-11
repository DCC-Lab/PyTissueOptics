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

    def testReflectNormalIncidence(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=zHat)
        intersect = FresnelIntersect(p.ez, s, distance=0)
        p.reflect(intersect)
        self.assertTrue( (p.ez+zHat).norm() < 1e-6)

    def testNoRefract(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=-zHat)
        s.indexInside = 1.0
        s.indexOutside = 1.0
        intersect = FresnelIntersect(zHat, s, distance=0)

        p = Photon(position=Vector(0,0,0), direction=zHat)
        p.refract(intersect)
        self.assertTrue( p.ez.isAlmostEqualTo(zHat))

    def testNormalRefract(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=-zHat)
        s.indexInside = 1.4
        s.indexOutside = 1.0
        p = Photon(position=Vector(0,0,0), direction=zHat)

        intersect = FresnelIntersect(p.ez, s, distance=0)

        p.refract(intersect)
        self.assertTrue( p.ez.isAlmostEqualTo(zHat))

    def testRefractIntoFrom(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        s.indexInside = 1.0
        s.indexOutside = 1.4

        for i in range(100):

            thetaIn = (i-50)*np.pi/4/50
            sinIn = np.sin(thetaIn)
            cosIn = np.cos(thetaIn)
            vIn = UnitVector(0,sinIn,cosIn)
            sinOut = np.sin(thetaIn)*s.indexInside/s.indexOutside
            cosOut = sqrt(1-sinOut*sinOut)
            vOut = UnitVector(0,sinOut, cosOut)

            p = Photon(position=Vector(0,0,0), direction=vIn)
            intersection = FresnelIntersect(p.ez, s, distance=0)

            p.refract(intersection)


            self.assertTrue( p.ez.isAlmostEqualTo(vOut), "ez: {0} vOut: {1} vIn {2}".format(p.ez, vOut, vIn))

    def testRefractOutOf(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        s.indexInside = 1.0
        s.indexOutside = 1.4

        for i in range(100):
            thetaIn = (i-50)*np.pi/2.1/50
            sinIn = np.sin(thetaIn)
            cosIn = np.cos(thetaIn)
            vIn = UnitVector(0,sinIn,cosIn)
            sinOut = np.sin(thetaIn)*s.indexInside/s.indexOutside

            p = Photon(position=Vector(0,0,0), direction=vIn)
            intersect = FresnelIntersect(p.ez, s, distance=0)

            if abs(sinOut) >= 1:
                continue
            thetaOut = np.arcsin(sinOut)
            cosOut = np.cos(thetaOut)

            vOut = UnitVector(0,sinOut, cosOut)
            p.refract(intersect)

            self.assertTrue( p.ez.isAlmostEqualTo(vOut), "ez: {0} vOut: {1} vIn {2}".format(p.ez, vOut, vIn))


if __name__ == '__main__':
    envtest.main()
