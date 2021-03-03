import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")

class TestVector(envtest.PyTissueTestCase):
    def testNullVector(self):
        vs = Vectors(N=1000)
        self.assertIsNotNone(vs)
        self.assertEqual(vs.count, 1000)

        for v in vs:        
            self.assertEqual(v.x, 0)
            self.assertEqual(v.y, 0)
            self.assertEqual(v.z, 0)

    def testSomeVectors(self):
        vs = Vectors(self.randomVectors(N=1000))
        self.assertIsNotNone(vs)
        self.assertEqual(vs.count, 1000)

        for v in vs:        
            self.assertTrue(abs(v.x) <= 1)
            self.assertTrue(abs(v.y) <= 1)
            self.assertTrue(abs(v.z) <= 1)

    def testSomeVectors(self):
        vs = Vectors(self.randomVectors(N=1000))
        self.assertEqual(len(vs), vs.count)

    def testAddVectors(self):
        v1 = Vectors(vectors=[Vector(1,1,1)]*1000)
        v2 = Vectors(vectors=[Vector(1,2,3)]*1000)

        vs = v1 + v2
        self.assertEqual(vs.count, 1000)
        for v in vs:        
            self.assertEqual(v.x, 2)
            self.assertEqual(v.y, 3)
            self.assertEqual(v.z, 4)

    def testSubVectors(self):
        v1 = Vectors(vectors=[Vector(1,1,1)]*1000)
        v2 = Vectors(vectors=[Vector(1,2,3)]*1000)

        vs = v1 - v2
        self.assertEqual(vs.count, 1000)
        for v in vs:        
            self.assertEqual(v.x, 0)
            self.assertEqual(v.y, -1)
            self.assertEqual(v.z, -2)

    def testEqualVectors(self):
        v1 = Vectors([oHat, xHat, yHat])
        v2 = Vectors([oHat, yHat, yHat])

        self.assertEqual(v1.isEqualTo(v2), [True, False, True])

    def testScaleVectors(self):
        v1 = Vectors([oHat, xHat, yHat])
        s = [1,2,3]

        self.assertTrue(v1*s == Vectors([oHat, 2*xHat, 3*yHat]))
        self.assertTrue(s*v1 == Vectors([oHat, 2*xHat, 3*yHat]))

    # def testGetItemVector(self):
    #     v = Vector(1,2,3)
    #     self.assertIsNotNone(v)
    #     self.assertEqual(v[0], 1)
    #     self.assertEqual(v[1], 2)
    #     self.assertEqual(v[2], 3)
    #     with self.assertRaises(ValueError):
    #         v[3]

    # def testVectorValueInit(self):
    #     v = Vector(1,2,3)
    #     self.assertIsNotNone(v)
    #     self.assertEqual(v.x, 1)
    #     self.assertEqual(v.y, 2)
    #     self.assertEqual(v.z, 3)

    # def testVectorCopy(self):
    #     constVector = Vector(1,2,3)
    #     v = Vector(constVector)
    #     self.assertIsNotNone(v)
    #     self.assertEqual(v.x, 1)
    #     self.assertEqual(v.y, 2)
    #     self.assertEqual(v.z, 3)

    #     constVector.x = 4
    #     self.assertEqual(v.x, 1)

    # def testInvalidInput(self):
    #     with self.assertRaises(ValueError):
    #         Vector("ouch")

    # def testConstVectors(self):
    #     self.assertTrue(oHat.isEqualTo(Vector(0,0,0)))
    #     self.assertTrue(xHat.isEqualTo(Vector(1,0,0)))
    #     self.assertTrue(yHat.isEqualTo(Vector(0,1,0)))
    #     self.assertTrue(zHat.isEqualTo(Vector(0,0,1)))

    def testEqualsVectors(self):
        v1 = Vectors([oHat, xHat, yHat])
        v2 = Vectors([oHat, yHat, yHat])
        v3 = Vectors([oHat, yHat, yHat])

        self.assertFalse(v1 == v2)
        self.assertTrue(v2 == v3)

    def testCanSetValuesVector(self):
        v = Vector()
        v.x = 1
        v.y = 2
        v.z = 3
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)

    # def testCannotSetConstVectors(self):      
    #     v = ConstVector(1,2,3)

    #     with self.assertRaises(RuntimeError):
    #         v.x = 1
    #     with self.assertRaises(RuntimeError):
    #         v.y = 2
    #     with self.assertRaises(RuntimeError):
    #         v.z = 3

    #     with self.assertRaises(RuntimeError):
    #         xHat.x = 2
    #     with self.assertRaises(RuntimeError):
    #         xHat.y = 2
    #     with self.assertRaises(RuntimeError):
    #         xHat.z = 2

    #     with self.assertRaises(RuntimeError):
    #         yHat.x = 2
    #     with self.assertRaises(RuntimeError):
    #         yHat.y = 2
    #     with self.assertRaises(RuntimeError):
    #         yHat.z = 2

    #     with self.assertRaises(RuntimeError):
    #         zHat.x = 2
    #     with self.assertRaises(RuntimeError):
    #         zHat.y = 2
    #     with self.assertRaises(RuntimeError):
    #         zHat.z = 2

    # def testCannotNormalizeConstVectors(self):      
    #     v = ConstVector(1,2,3)

    #     with self.assertRaises(RuntimeError):
    #         v.normalize()

    def testVectorIsNull(self):
        vs = Vectors([oHat, xHat, yHat])
        self.assertEqual(vs.isNull, [True, False, False])

    def testVectorIsUnitary(self):
        vs = Vectors([oHat, xHat, yHat])
        self.assertEqual(vs.isUnitary, [False, True, True])

    # def testNormalizedCrossedProduct(self):
    #     v1 = Vector(1,2,3)
    #     v2 = v1.anyPerpendicular()
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), 1.0, 6)

    #     v1 = Vector(1,0,0)
    #     v2 = Vector(1,1,0)
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), np.sin(np.pi/4), 6)

    #     v1 = Vector(2,0,0)
    #     v2 = Vector(2,2,0)
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), np.sin(np.pi/4), 6)

    #     v1 = Vector(0,0,1)
    #     v2 = Vector(0,1,1)
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), np.sin(np.pi/4), 6)

    #     v1 = Vector(0,1,0)
    #     v2 = Vector(0,1,1)
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), np.sin(np.pi/4), 6)

    #     v1 = Vector(0,1,0)
    #     v2 = Vector(0,0,0)
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), 0, 6)

    # def testOperators(self):
    #     v1 = Vector(1,2,3)
    #     self.assertTrue( (v1*2).isAlmostEqualTo(Vector(2,4,6)))
    #     self.assertTrue( (2*v1).isAlmostEqualTo(Vector(2,4,6)))
    #     self.assertTrue( (v1/2.0).isAlmostEqualTo(Vector(0.5,1,1.5)))
    #     self.assertTrue( (v1+v1+v1).isAlmostEqualTo(Vector(3,6,9)))
    #     self.assertTrue( (v1-2*v1).isAlmostEqualTo(Vector(-1,-2,-3)))
    #     self.assertTrue( (-v1).isAlmostEqualTo(Vector(-1,-2,-3)))

    # def testRotateAround(self):
    #     v1 = Vector(1,2,3)
    #     axis = v1.anyUnitaryPerpendicular()
    #     v1.rotateAround(axis, 0.5)
    #     v2 = Vector(1,2,3)
    #     self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), np.sin(0.5), 6)

    #     for i in range(1000):
    #         v1 = self.randomVector()
    #         v2 = Vector(v1)
    #         axis = v1.anyUnitaryPerpendicular()
    #         angle = 2*np.pi*random.random()
    #         v1.rotateAround(axis, angle)
    #         self.assertAlmostEqual(v1.normalizedCrossProduct(v2).abs(), abs(np.sin(angle)), 6)

    # def testAngleWith(self):
    #     for i in range(1000):
    #         v1 = self.randomVector()
    #         v2 = Vector(v1)
    #         axis = v1.anyUnitaryPerpendicular()
    #         angle = np.pi*(random.random()*2-1)
    #         v1.rotateAround(axis, angle)
    #         self.assertAlmostEqual(v1.angleWith(v2, axis), -angle, 6)

    # def testVectorIsParallel(self):     
    #     v1 = Vector(1,2,3)
    #     v2 = Vector(2,4,6)

    #     self.assertTrue(v1.isParallelTo(v2))
    #     self.assertFalse(v1.isParallelTo(xHat))
    #     self.assertFalse(v1.isParallelTo(yHat))
    #     self.assertFalse(v1.isParallelTo(zHat))

    # def testVectorIsAlmostParallel(self):     
    #     v1 = Vector(1,2,3)
    #     v3 = Vector(1,2,3.0000001)
    #     self.assertTrue(v1.isParallelTo(v3))

    # def testVectorIsAlmostNotParallel(self):     
    #     v1 = Vector(1,2,3)
    #     v3 = Vector(1,2,3.0001)
    #     self.assertFalse(v1.isParallelTo(v3))

    # def testVectorIsPerpendicular(self):     
    #     self.assertTrue(xHat.isPerpendicularTo(yHat))
    #     self.assertTrue(xHat.isPerpendicularTo(zHat))
    #     self.assertFalse(xHat.isPerpendicularTo(xHat))

    #     self.assertTrue(yHat.isPerpendicularTo(zHat))
    #     self.assertTrue(yHat.isPerpendicularTo(xHat))
    #     self.assertFalse(yHat.isPerpendicularTo(yHat))

    #     self.assertTrue(zHat.isPerpendicularTo(xHat))
    #     self.assertTrue(zHat.isPerpendicularTo(yHat))
    #     self.assertFalse(zHat.isPerpendicularTo(zHat))

    # def testManyPerpendicularVectors(self):
    #     for i in range(10000):
    #         v1 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
    #         v2 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
    #         self.assertTrue(v1.cross(v2).isPerpendicularTo(v1))
    #         self.assertTrue(v1.cross(v2).isPerpendicularTo(v2))

    # def testAngle(self):
    #     self.assertEqual(xHat.angleWith(yHat, zHat), np.pi/2)
    #     self.assertEqual(yHat.angleWith(xHat, zHat), -np.pi/2)
    #     self.assertEqual(xHat.angleWith(yHat, -zHat),-np.pi/2)

    # def testQuadrantAngle(self):
    #     plus45 = xHat + yHat
    #     plus135 = -xHat + yHat
    #     minus45 = xHat - yHat
    #     minus135 = -xHat - yHat

    #     self.assertAlmostEqual(xHat.angleWith(plus45, zHat), np.pi/4, 6)
    #     self.assertAlmostEqual(xHat.angleWith(plus135, zHat), 3*np.pi/4, 6)

    #     self.assertAlmostEqual(xHat.angleWith(minus45, zHat), -np.pi/4, 6)
    #     self.assertAlmostEqual(xHat.angleWith(minus135, zHat), -3*np.pi/4, 6)

    #     self.assertAlmostEqual(xHat.angleWith(plus45, -zHat), -np.pi/4, 6)
    #     self.assertAlmostEqual(xHat.angleWith(plus135, -zHat), -3*np.pi/4, 6)

    #     self.assertAlmostEqual(xHat.angleWith(minus45, -zHat), np.pi/4, 6)
    #     self.assertAlmostEqual(xHat.angleWith(minus135, -zHat), 3*np.pi/4, 6)

    #     self.assertAlmostEqual(xHat.angleWith(-xHat, zHat), -np.pi, 6)

    # def testScaledSum(self):
    #     v1 = Vector(1,2,3)
    #     v2 = Vector(4,5,6)
    #     s  = 2
    #     v3 = Vector(9, 12, 15)
    #     self.assertEqual(Vector.fromScaledSum(v1, v2, s), v3)
    #     self.assertEqual(v1.addScaled(v2, s), v3)

    # def testAnyPerpendicular(self):
    #     vectors = (Vector(1,2,3), Vector(-1,-2-3), xHat, yHat, zHat )
    #     for v in vectors:
    #         self.assertTrue(v.anyPerpendicular().isPerpendicularTo(v))

    #     self.assertIsNone(oHat.anyPerpendicular())
    #     self.assertIsNone(oHat.anyPerpendicular())

    # def testIsInKnownPlane(self):
    #     v = Vector(1,2,3)
    #     self.assertTrue(v.isInXYPlane(atZ=3))
    #     self.assertTrue(v.isInYZPlane(atX=1))
    #     self.assertTrue(v.isInZXPlane(atY=2))

    #     self.assertFalse(v.isInXYPlane(atZ=3.1))
    #     self.assertFalse(v.isInYZPlane(atX=1.1))
    #     self.assertFalse(v.isInZXPlane(atY=2.1))

    # def testIsInSomePlane(self):
    #     v = Vector(1,1,0)
    #     self.assertTrue(v.isInPlane(origin=Vector(0,0,0), normal=zHat))
    #     self.assertFalse(v.isInPlane(origin=Vector(1,1,1), normal=zHat))

    # def testRotationsUnitVectors(self):
    #     for i in range(10000):
    #         v1 = UnitVector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
    #         v2 = UnitVector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
    #         axis = v1.cross(v2)
    #         angle = v1.angleWith(v2, axis)
    #         v1.rotateAround(axis, angle)

    #         self.assertTrue( (v1-v2).abs() < 1e-6)
    #         self.assertTrue( v1.isUnitary)
    #         self.assertTrue( v2.isUnitary)

    #     self.assertTrue(Vector(1,0,0).rotateAround(zHat, np.pi/2).isAlmostEqualTo( yHat))
    #     self.assertTrue(Vector(0,1,0).rotateAround(xHat, np.pi/2).isAlmostEqualTo( zHat))
    #     self.assertTrue(Vector(0,0,1).rotateAround(yHat, np.pi/2).isAlmostEqualTo( xHat))
            

    # def testRotationsVectors(self):
    #     for i in range(10000):
    #         v1 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
    #         v2 = Vector(random.random()*2-1, random.random()*2-1, random.random()*2-1)
    #         axis = v1.cross(v2)
    #         angle = v1.angleWith(v2, axis)
    #         v1.rotateAround(axis, angle)

    #         self.assertTrue( v1.dot(v2) >= 0)
    #         self.assertTrue( v1.cross(v2).abs() < 1e-6)
    #         self.assertTrue( v1.isParallelTo(v2) )

    # def testPlaneOfIncidence(self):
    #     ez = Vector(0,1,1).normalized()
    #     plane = ez.planeOfIncidence(normal=zHat)
    #     self.assertTrue(plane.isParallelTo(xHat))
    #     self.assertTrue(plane.isUnitary)

    #     ez = Vector(0,1,1).normalized()
    #     plane = ez.planeOfIncidence(normal=-zHat)
    #     self.assertTrue(plane.isParallelTo(xHat))
    #     self.assertTrue(plane.isUnitary)

    #     ez = Vector(0,-1,1).normalized()
    #     plane = ez.planeOfIncidence(normal=zHat)
    #     self.assertTrue(plane.isParallelTo(xHat))
    #     self.assertTrue(plane.isUnitary)

    #     ez = Vector(0,1,1) # unnormalized
    #     plane = ez.planeOfIncidence(normal=Vector(0,0,10))
    #     self.assertTrue(plane.isParallelTo(xHat))
    #     self.assertTrue(plane.isUnitary)

    #     plane = zHat.planeOfIncidence(normal=-zHat)
    #     self.assertTrue(plane.isPerpendicularTo(zHat))
    #     self.assertTrue(plane.isUnitary)

    #     plane = zHat.planeOfIncidence(normal=-zHat)
    #     self.assertTrue(plane.isPerpendicularTo(zHat))
    #     self.assertTrue(plane.isUnitary)

    #     plane = zHat.planeOfIncidence(normal=zHat)
    #     self.assertTrue(plane.isPerpendicularTo(zHat))
    #     self.assertTrue(plane.isUnitary)

    # def testAngleOfIncidence(self):
    #     ez = Vector(0,1,1).normalized()
    #     surfaceNormal = Vector(0,0,1)
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(normal=surfaceNormal)
    #     self.assertAlmostEqual(ez.angleWith(surfaceNormal, planeNormal), np.pi/4,6)

    #     surfaceNormal = Vector(0,0,-1)
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(normal=surfaceNormal)
    #     self.assertAlmostEqual(ez.angleWith(actualNormal, planeNormal), np.pi/4,6)

    #     ez = Vector(0,-1,1).normalized()
    #     surfaceNormal = zHat
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(normal=surfaceNormal)
    #     self.assertAlmostEqual(ez.angleWith(actualNormal, planeNormal), np.pi/4,6)

    #     ez = Vector(0,1,1).normalized()
    #     surfaceNormal = zHat
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(surfaceNormal)
    #     self.assertAlmostEqual(angle, np.pi/4,6)
    #     self.assertTrue(planeNormal.isUnitary)

    #     ez = Vector(0,-1,1).normalized()
    #     surfaceNormal = zHat
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(surfaceNormal)
    #     self.assertAlmostEqual(angle, np.pi/4,6)
    #     self.assertTrue(planeNormal.isUnitary)

    #     ez = zHat
    #     surfaceNormal = zHat
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(surfaceNormal)
    #     self.assertAlmostEqual(angle, 0,6)
    #     self.assertTrue(planeNormal.isUnitary)

    #     ez = Vector(0,0,1)
    #     surfaceNormal = Vector(0,0,1)
    #     angle, planeNormal, actualNormal = ez.angleOfIncidence(surfaceNormal)
    #     self.assertAlmostEqual(angle, 0,6)
    #     self.assertTrue(planeNormal.isUnitary)

    def randomVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2-1
        return Vector(x,y,z)

    def randomVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( Vector(x,y,z) )
        return vectors

    def randomUnitVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( UnitVector(x,y,z) )
        return vectors


if __name__ == '__main__':
    envtest.main()
