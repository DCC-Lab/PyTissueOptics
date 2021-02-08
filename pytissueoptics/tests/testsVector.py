import envtest # modifies path
from pytissueoptics import *
import numpy as np

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


if __name__ == '__main__':
    envtest.main()
