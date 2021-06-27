import envtest  # modifies path
from pytissueoptics import *
from numpy import linspace, pi, sqrt, polyfit
from pytissueoptics.geometry import Sphere
import matplotlib.pyplot as plt

inf = float("+inf")
class TestSphere(envtest.PyTissueTestCase):

    def testSphere(self):
        self.assertIsNotNone(Sphere(radius=10, material=None))

    def testOrigin(self):
        sphere = Sphere(radius=10, material=None)
        self.assertEqual(sphere.origin,oHat)

    def testGeometryContains(self):
        sphere = Sphere(radius=10, material=None)
        for i in range(1000):
            v = self.randomVector()
            self.assertTrue(sphere.contains(v))

    def testSurfacesContain(self):
        sphere = Sphere(radius=1, material=None)
        for i in range(1000):
            v = self.randomUnitVector()

            s1, s2 = sphere.surfaces
            isOnS1,_,_ = s1.contains(v)
            isOnS2,_,_ = s2.contains(v)
            self.assertTrue( isOnS1 or isOnS2 )

        for i in range(1000):
            v = self.randomUnitVector()*0.99

            s1, s2 = sphere.surfaces
            isOnS1,_,_ = s1.contains(v)
            isOnS2,_,_ = s2.contains(v)
            self.assertFalse( isOnS1 or isOnS2 )

        for i in range(1000):
            v = self.randomUnitVector()*1.01

            s1, s2 = sphere.surfaces
            isOnS1,_,_ = s1.contains(v)
            isOnS2,_,_ = s2.contains(v)
            self.assertFalse( isOnS1 or isOnS2 )

if __name__ == '__main__':
    envtest.main()
