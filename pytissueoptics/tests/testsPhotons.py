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

    def testPlaneOfIncidence(self):
        zHat.planeOfIncidence()

    # def testReflectInto(self):
    #     s=Surface(origin=oHat, a=xHat, b=yHat, normal=zHat)
    #     self.assertEqual(s.normal, zHat)

    #     p = Photon(position=Vector(0,0,0), direction=zHat)
    #     planeOfIncidenceNormal = p.ez.normalizedCrossProduct(s.normal)
    #     print(planeOfIncidenceNormal)

    #     p.reflect(s)
    #     print(p.ez)
    #     self.assertTrue(p.ez == -zHat)

if __name__ == '__main__':
    envtest.main()
