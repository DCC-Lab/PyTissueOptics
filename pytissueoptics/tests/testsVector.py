import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")

class TestVector(envtest.PyTissueTestCase):

    def testNullVector(self):
        v = Vector()
        self.assertIsNotNone(v)
        self.assertEqual(v.x, 0)
        self.assertEqual(v.y, 0)
        self.assertEqual(v.z, 0)

    def testVectorValueInit(self):
        v = Vector(1,2,3)
        self.assertIsNotNone(v)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    def testVectorCopy(self):
        constVector = Vector(1,2,3)
        v = Vector(constVector)
        self.assertIsNotNone(v)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

        constVector.x = 4
        self.assertEqual(v.x, 1)

    # def testVectorInitArray(self):
    #     v = Vector(np.ndarray((1,2,3)))
    #     self.assertIsNotNone(v)
    #     self.assertEqual(v.x, 1)
    #     self.assertEqual(v.y, 2)
    #     self.assertEqual(v.z, 3)

    def testConstVectors(self):
        self.assertTrue(oHat.isEqualTo(Vector(0,0,0)))
        self.assertTrue(xHat.isEqualTo(Vector(1,0,0)))
        self.assertTrue(yHat.isEqualTo(Vector(0,1,0)))
        self.assertTrue(zHat.isEqualTo(Vector(0,0,1)))

    def testCanSetValuesVector(self):
        v = Vector()
        v.x = 1
        v.y = 2
        v.z = 3
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    def testCannotSetConstVectors(self):      
        with self.assertRaises(RuntimeError):
            xHat.x = 2
        with self.assertRaises(RuntimeError):
            xHat.y = 2
        with self.assertRaises(RuntimeError):
            xHat.z = 2

        with self.assertRaises(RuntimeError):
            yHat.x = 2
        with self.assertRaises(RuntimeError):
            yHat.y = 2
        with self.assertRaises(RuntimeError):
            yHat.z = 2

        with self.assertRaises(RuntimeError):
            zHat.x = 2
        with self.assertRaises(RuntimeError):
            zHat.y = 2
        with self.assertRaises(RuntimeError):
            zHat.z = 2

    def testVectorIsParallel(self):     
        v1 = Vector(1,2,3)
        v2 = Vector(2,4,6)

        self.assertTrue(v1.isParallelTo(v2))
        self.assertFalse(v1.isParallelTo(xHat))
        self.assertFalse(v1.isParallelTo(yHat))
        self.assertFalse(v1.isParallelTo(zHat))

    def testVectorIsAlmostParallel(self):     
        v1 = Vector(1,2,3)
        v3 = Vector(1,2,3.0000001)
        self.assertTrue(v1.isParallelTo(v3))

    def testVectorIsAlmostNotParallel(self):     
        v1 = Vector(1,2,3)
        v3 = Vector(1,2,3.0001)
        self.assertFalse(v1.isParallelTo(v3))

    def testVectorIsPerpendicular(self):     
        self.assertTrue(xHat.isPerpendicularTo(yHat))
        self.assertTrue(xHat.isPerpendicularTo(zHat))
        self.assertFalse(xHat.isPerpendicularTo(xHat))

        self.assertTrue(yHat.isPerpendicularTo(zHat))
        self.assertTrue(yHat.isPerpendicularTo(xHat))
        self.assertFalse(yHat.isPerpendicularTo(yHat))

        self.assertTrue(zHat.isPerpendicularTo(xHat))
        self.assertTrue(zHat.isPerpendicularTo(yHat))
        self.assertFalse(zHat.isPerpendicularTo(zHat))

    def testManyPerpendicularVectors(self):
        for i in range(100000):
            v1 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
            v2 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
            self.assertTrue(v1.cross(v2).isPerpendicularTo(v1))
            self.assertTrue(v1.cross(v2).isPerpendicularTo(v2))

    def testAngle(self):
        self.assertEqual(xHat.angleWith(yHat, zHat), np.pi/2)
        self.assertEqual(yHat.angleWith(xHat, zHat), -np.pi/2)
        self.assertEqual(xHat.angleWith(yHat, -zHat),-np.pi/2)

    def testQuadrantAngle(self):
        plus45 = xHat + yHat
        plus135 = -xHat + yHat
        minus45 = xHat - yHat
        minus135 = -xHat - yHat

        self.assertAlmostEqual(xHat.angleWith(plus45, zHat), np.pi/4, 6)
        self.assertAlmostEqual(xHat.angleWith(plus135, zHat), 3*np.pi/4, 6)

        self.assertAlmostEqual(xHat.angleWith(minus45, zHat), -np.pi/4, 6)
        self.assertAlmostEqual(xHat.angleWith(minus135, zHat), -3*np.pi/4, 6)

        self.assertAlmostEqual(xHat.angleWith(plus45, -zHat), -np.pi/4, 6)
        self.assertAlmostEqual(xHat.angleWith(plus135, -zHat), -3*np.pi/4, 6)

        self.assertAlmostEqual(xHat.angleWith(minus45, -zHat), np.pi/4, 6)
        self.assertAlmostEqual(xHat.angleWith(minus135, -zHat), 3*np.pi/4, 6)

        self.assertAlmostEqual(xHat.angleWith(-xHat, zHat), -np.pi, 6)

    def testScaledSum(self):
        v1 = Vector(1,2,3)
        v2 = Vector(4,5,6)
        s  = 2
        v3 = Vector(9, 12, 15)
        self.assertEqual(Vector.fromScaledSum(v1, v2, s), v3)
        self.assertEqual(v1.addScaled(v2, s), v3)

    def testAnyPerpendicular(self):
        v1 = Vector(1,2,3)
        v2 = Vector(-4,-5,-6)

        self.assertTrue(v1.anyPerpendicular().isPerpendicularTo(v1))
        self.assertTrue(v2.anyPerpendicular().isPerpendicularTo(v2))
        self.assertIsNone(oHat.anyPerpendicular())


if __name__ == '__main__':
    envtest.main()
