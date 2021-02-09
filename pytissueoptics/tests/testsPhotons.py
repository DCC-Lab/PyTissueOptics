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

    # def testReflectInto(self):
    #     s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)

    #     p = Photon(position=Vector(0,0,0), direction=zHat)
    #     print(p.ez)
    #     p.reflect(s)
    #     print(p.ez)
    #     self.assertTrue(p.ez == -zHat)

    def testReflectNormalIncidence(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=zHat)

        thetaIn, planeOfIncidenceNormal = p.ez.angleOfIncidence(s.normal)
        p.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)
        self.assertAlmostEqual((p.ez - -zHat).norm(), 0, 6)

    def testReflectPlus45Incidence(self):
        s = Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
        p = Photon(position=Vector(0,0,0), direction=Vector(0,1,1).normalized())

        thetaIn, planeOfIncidenceNormal = p.ez.angleOfIncidence(s.normal)
        p.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)
        self.assertAlmostEqual((p.ez - Vector(0,1,-1).normalized()).norm(), 0, 6)

if __name__ == '__main__':
    envtest.main()
