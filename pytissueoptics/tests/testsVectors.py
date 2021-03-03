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

    def testDivideVectors(self):
        v1 = Vectors([oHat, xHat, yHat])
        s = [1,2,3]

        self.assertTrue(v1/s == Vectors([oHat, xHat/2, yHat/3]))

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

    def testVectorNegative(self):
        vs = Vectors([oHat, xHat, yHat])
        self.assertEqual(-vs, Vectors([-oHat, -xHat, -yHat]))

    def testVectorAlmostEqual(self):
        v1 = Vectors([oHat, xHat, yHat])
        v2 = Vectors([Vector(0,0,1), Vector(1.0000001,0,0), Vector(0,1.01,0)])
        self.assertEqual(v1.isAlmostEqualTo(v2,epsilon=1e-6), [False, True, False])

    def testVectorParallel(self):
        v1 = Vectors([oHat, xHat, yHat])
        v2 = Vectors([oHat, xHat, xHat])
        self.assertEqual(v1.isParallelTo(v2,epsilon=1e-6), [True, True, False])

    def testVectorPerpendicular2(self):
        v1 = Vectors([zHat, xHat, yHat])
        v2 = Vectors([Vector(1,1,1), zHat, xHat])
        self.assertEqual(v1.isPerpendicularTo(v2,epsilon=1e-6), [False, True, True])

    def testCrossProduct(self):
        v1 = Vectors([xHat, yHat, zHat,xHat, yHat, zHat])
        v2 = Vectors([yHat, zHat, xHat,xHat, yHat, zHat])
        v3 = Vectors([zHat, xHat, yHat,oHat, oHat, oHat])
        self.assertEqual(v1.cross(v2), v3)

    def testNormalizedCrossProduct(self):
        v1 = Vectors([xHat, yHat, zHat,xHat, yHat, zHat])
        v2 = Vectors([yHat, zHat, 2*xHat,3*xHat, yHat, zHat])
        v3 = Vectors([zHat, xHat, yHat,oHat, oHat, oHat])
        self.assertEqual(v1.normalizedCrossProduct(v2), v3)

    def testDotProduct(self):
        v1 = Vectors([xHat, yHat, zHat])
        v2 = Vectors([xHat, zHat, 2*zHat])
        self.assertEqual(v1.dot(v2), [1,0,2])

    def testAngle(self):
        v1 = Vectors([xHat, yHat, zHat])
        v2 = Vectors([xHat, zHat, Vector(0,1,1)])
        axes = Vectors([yHat, xHat, xHat])
        actual = v1.angleWith(v2, axes)
        angles= [0, np.pi/2, -np.pi/4]
        for i in range(3):
            self.assertAlmostEqual(angles[i], actual[i] )

    def testNormalizedDotProduct(self):
        v1 = Vectors([xHat, yHat, zHat])
        v2 = Vectors([xHat, zHat, 2*zHat])
        self.assertEqual(v1.normalizedDotProduct(v2), [1,0,1])

    def testRotateAround(self):
        v1 = Vectors([xHat, yHat, zHat])
        axis = Vectors([zHat, xHat, yHat])
        v3 = Vectors(v1)
        angle = [np.pi/2, np.pi/2, np.pi/2]

        v1.rotateAround(axis, angle)

        actual = v3.angleWith(v1, axis)

        for i in range(3):
            self.assertAlmostEqual(angle[i], actual[i],3,"Error at {0}".format(i))

    def testRotatedAround(self):
        v1 = Vectors([xHat, yHat, zHat])
        axis = Vectors([zHat, xHat, yHat])
        angle = [np.pi/2, np.pi/2, np.pi/2]

        v3 = v1.rotatedAround(axis, angle)

        actual = v1.angleWith(v3, axis)

        for i in range(3):
            self.assertAlmostEqual(angle[i], actual[i],3,"Error at {0}".format(i))

    def testPlaneAndAngleOfIncidence(self):
        v1 = Vectors([Vector(0,1,1), Vector(0,-1,1), Vector(1,0,1)])
        normal = Vectors([zHat, zHat, zHat])

        angles = [np.pi/4, np.pi/4, np.pi/4]
        expected = Vectors([xHat, -xHat, -yHat])
        actual = v1.planeOfIncidence(normal)
        actualAngles, actualPlanes, correctedNormal = v1.angleOfIncidence(normal)

        for i in range(3):
            self.assertTrue(actual[i].isAlmostEqualTo(expected[i]))
            self.assertAlmostEqual(actualAngles[i], angles[i])

    def testNormAbs(self):
        v1 = Vectors([Vector(1,1,1), Vector(1,0,1),Vector(0,1,1)]) 
        self.assertEqual(v1.norm(), [3,2,2])
        self.assertEqual(v1.abs(), [np.sqrt(3),np.sqrt(2),np.sqrt(2)])

    def testNormalize(self):
        v1 = Vectors([Vector(1,1,1), Vector(1,0,1),Vector(0,1,1)]) 
        v2 = v1.normalized()
        self.assertTrue(np.array(v2.isUnitary).all())

        v1.normalize()
        self.assertTrue(np.array(v1.isUnitary).all())

    def testScaledSum(self):
        v1 = Vectors([Vector(1,2,3), Vector(4,5,6),Vector(7,8,9)]) 
        v2 = Vectors([Vector(1,2,3), Vector(4,5,6),Vector(7,8,9)]) 
        s  = [1,2,3]
        v3 = Vectors([Vector(2,4,6), Vector(12,15,18),Vector(28,32,36)])
        self.assertEqual(Vectors.fromScaledSum(v1, v2, s), v3)
        self.assertEqual(v1.addScaled(v2, s), v3)

    def testAnyPerpendicular(self):
        vectors = Vectors([Vector(1,2,3), Vector(-1,-2-3), xHat, yHat, zHat])
        for v in vectors:
            self.assertTrue(v.anyPerpendicular().isPerpendicularTo(v))

        self.assertIsNone(oHat.anyPerpendicular())
        self.assertIsNone(oHat.anyPerpendicular())

    def testAnyPerpendicular(self):
        vectors = Vectors(vectors=self.randomVectors(N=10000))
        perp = vectors.anyPerpendicular()
        isPerp = vectors.isPerpendicularTo(perp)

        self.assertTrue(np.array(isPerp).all())

    def testAnyUnitaryPerpendicular(self):
        vectors = Vectors(vectors=self.randomVectors(N=10000))
        perp = vectors.anyUnitaryPerpendicular()
        isPerp = vectors.isPerpendicularTo(perp, epsilon=1e-5)
        isUnitary = perp.isUnitary

        self.assertTrue(np.array(isPerp).all())
        self.assertTrue(np.array(isUnitary).all())

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

    def randomVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2-1
        return Vector(x,y,z)

    def randomAngles(self, N):
        scalar = []
        for i in range(N):
            scalar.append( (random.random()-0.5)*2*np.pi)
        return scalar

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
