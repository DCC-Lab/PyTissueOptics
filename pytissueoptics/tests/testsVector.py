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

    def testInvalidInput(self):
        with self.assertRaises(ValueError):
            Vector("ouch")

    def testConstVectors(self):
        self.assertTrue(oHat.isEqualTo(Vector(0,0,0)))
        self.assertTrue(xHat.isEqualTo(Vector(1,0,0)))
        self.assertTrue(yHat.isEqualTo(Vector(0,1,0)))
        self.assertTrue(zHat.isEqualTo(Vector(0,0,1)))

    def testEqualVectors(self):
        self.assertEqual(oHat, Vector(0,0,0))
        self.assertEqual(xHat, Vector(1,0,0))
        self.assertEqual(yHat, Vector(0,1,0))
        self.assertEqual(zHat, Vector(0,0,1))

        self.assertNotEqual(Vector(1,2,3), Vector(0,2,3))
        self.assertNotEqual(Vector(1,2,3), Vector(1,0,3))
        self.assertNotEqual(Vector(1,2,3), Vector(1,2,0))


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

    def testVectorIsNull(self):
        self.assertTrue(oHat.isNull)

    def testVectorIsUnitary(self):
        self.assertTrue(xHat.isUnitary)
        self.assertTrue(yHat.isUnitary)
        self.assertTrue(zHat.isUnitary)

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
        for i in range(10000):
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
        vectors = (Vector(1,2,3), Vector(-1,-2-3), xHat, yHat, zHat )
        for v in vectors:
            self.assertTrue(v.anyPerpendicular().isPerpendicularTo(v))

        self.assertIsNone(oHat.anyPerpendicular())

    def testIsInKnownPlane(self):
        v = Vector(1,2,3)
        self.assertTrue(v.isInXYPlane(atZ=3))
        self.assertTrue(v.isInYZPlane(atX=1))
        self.assertTrue(v.isInZXPlane(atY=2))

        self.assertFalse(v.isInXYPlane(atZ=3.1))
        self.assertFalse(v.isInYZPlane(atX=1.1))
        self.assertFalse(v.isInZXPlane(atY=2.1))

    def testIsInSomePlane(self):
        v = Vector(1,1,0)
        self.assertTrue(v.isInPlane(origin=Vector(0,0,0), normal=zHat))
        self.assertFalse(v.isInPlane(origin=Vector(1,1,1), normal=zHat))

    def testRotationsUnitVectors(self):
        for i in range(10000):
            v1 = UnitVector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
            v2 = UnitVector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
            axis = v1.cross(v2)
            angle = v1.angleWith(v2, axis)
            v1.rotateAround(axis, angle)

            self.assertTrue( (v1-v2).abs() < 1e-6)

        self.assertEqual(Vector(1,0,0).rotateAround(zHat, np.pi/2), yHat)
        self.assertEqual(Vector(0,1,0).rotateAround(xHat, np.pi/2), zHat)
        self.assertEqual(Vector(0,0,1).rotateAround(yHat, np.pi/2), xHat)

    def testRotationsUnitVectors(self):
        for i in range(10000):
            v1 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
            v2 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
            axis = v1.cross(v2)
            angle = v1.angleWith(v2, axis)
            v1.rotateAround(axis, angle)

            self.assertTrue( v1.dot(v2) >= 0)
            self.assertTrue( v1.cross(v2).abs() < 1e-6)
            self.assertTrue( v1.isParallelTo(v2) )

    def testPlaneOfIncidence(self):
        ez = Vector(0,1,1).normalized()
        plane = ez.planeOfIncidence(normal=zHat)
        self.assertTrue(plane.isParallelTo(xHat))
        self.assertTrue(plane.isUnitary)

        ez = Vector(0,1,1).normalized()
        plane = ez.planeOfIncidence(normal=-zHat)
        self.assertTrue(plane.isParallelTo(xHat))
        self.assertTrue(plane.isUnitary)

        plane = zHat.planeOfIncidence(normal=-zHat)
        self.assertTrue(plane.isPerpendicularTo(zHat))
        self.assertTrue(plane.isUnitary)

        plane = zHat.planeOfIncidence(normal=-zHat)
        self.assertTrue(plane.isPerpendicularTo(zHat))
        self.assertTrue(plane.isUnitary)

        plane = zHat.planeOfIncidence(normal=zHat)
        self.assertTrue(plane.isPerpendicularTo(zHat))
        self.assertTrue(plane.isUnitary)

if __name__ == '__main__':
    envtest.main()
