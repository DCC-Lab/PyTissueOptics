import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")

class TestVectors(envtest.PyTissueTestCase):
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

    def testAddVectorsWithMask(self):
        v1 = Vectors(vectors=[Vector(1,1,1)]*3)
        v1.selected = [False, True, True]
        v2 = Vectors(vectors=[Vector(1,2,3)]*3)

        vs = v1 + v2
        self.assertEqual(vs.count, 3)
        self.assertEqual(vs, [Vector(1,1,1), Vector(2,3,4), Vector(2,3,4)])

    def testSubVectors(self):
        v1 = Vectors(vectors=[Vector(1,1,1)]*1000)
        v2 = Vectors(vectors=[Vector(1,2,3)]*1000)

        vs = v1 - v2
        self.assertEqual(vs.count, 1000)
        for v in vs:        
            self.assertEqual(v.x, 0)
            self.assertEqual(v.y, -1)
            self.assertEqual(v.z, -2)

    def testSubVectorsWithMask(self):
        v1 = Vectors(vectors=[Vector(1,1,1)]*3)
        v1.selected = [False, True, True]
        v2 = Vectors(vectors=[Vector(1,2,3)]*3)

        vs = v1 - v2
        self.assertEqual(vs.count, 3)
        self.assertEqual(vs, [Vector(1,1,1), Vector(0,-1,-2), Vector(0,-1,-2)])

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

    def testVectorIsNull(self):
        vs = Vectors([oHat, xHat, yHat])
        self.assertEqual(vs.isNull, [True, False, False])

    def testVectorIsUnitary(self):
        vs = Vectors([oHat, xHat, yHat])
        self.assertEqual(vs.isUnitary, [False, True, True])

    def testVectorIsUnitaryWithMask(self):
        vs = Vectors([oHat, xHat, yHat])
        vs.selected = [False, True, False]
        self.assertEqual(vs.isUnitary, [False, True, False])

    def testVectorNegative(self):
        vs = Vectors([oHat, xHat, yHat])
        self.assertEqual(-vs, Vectors([-oHat, -xHat, -yHat]))

    def testVectorNegativeWithMask(self):
        vs = Vectors([oHat, xHat, yHat])
        vs.selected = [False, True, False]
        self.assertEqual(-vs, Vectors([oHat, -xHat, yHat]))

    def testVectorAlmostEqual(self):
        v1 = Vectors([oHat, xHat, yHat])
        v2 = Vectors([Vector(0,0,1), Vector(1.0000001,0,0), Vector(0,1.01,0)])
        self.assertEqual(v1.isAlmostEqualTo(v2,epsilon=1e-6), [False, True, False])

    def testVectorAlmostEqualWithMask(self):
        v1 = Vectors([oHat, xHat, yHat])
        v1.selected = [False, True, True]
        v2 = Vectors([Vector(0,0,1), Vector(1.0000001,0,0), Vector(0,1.01,0)])
        r = v1.isAlmostEqualTo(v2,epsilon=1e-6)
        self.assertEqual(r, [False, True, False])

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
        normal = Vectors([zHat, zHat, -zHat])

        expectedAngles = [np.pi/4, np.pi/4, np.pi/4]
        expectedPlanes = Vectors([xHat, -xHat, -yHat])
        actualAngles, actualPlanes, correctedNormal = v1.angleOfIncidence(normal)
        print(actualPlanes.v)
        for i in range(3):
            self.assertTrue(actualPlanes[i].isAlmostEqualTo(expectedPlanes[i]))
            self.assertAlmostEqual(actualAngles[i], expectedAngles[i])

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

    def testNormalizeWithMask(self):
        v1 = Vectors([Vector(1,1,1), Vector(1,0,1),Vector(0,1,1)])
        v1.selected = [False, True, True]
        v2 = v1.normalized()

        v1.normalize()
        self.assertEqual(v1[0], Vector(1,1,1))

    def testScaledSum(self):
        v1 = Vectors([Vector(1,2,3), Vector(4,5,6),Vector(7,8,9)])
        v1.selected = [False, True, True]
        v2 = Vectors([Vector(1,2,3), Vector(4,5,6),Vector(7,8,9)]) 
        s  = [1,2,3]
        v3 = Vectors([Vector(1,2,3), Vector(12,15,18),Vector(28,32,36)])
        r = Vectors.fromScaledSum(v1, v2, s)
        self.assertEqual(r, v3)
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

class TestNumpyVectors(envtest.PyTissueTestCase):

    def testInitWithList(self):
        pass

    def testInitWithNumpyArray(self):
        pass

    def testCheckInitTypeFloat64(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        self.assertEqual(np.float64, type(vecs[0][0]))

    def testGetItem(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        r = vecs[0]
        r = np.equal(r, [1, 1, 1])
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testSetItem(self):
        pass

    def testAdd(self):
        vecs = NumpyVectors([[1, 1, 1], [0, 1, 0], [-1, 0, 0]])
        r = vecs + 1
        r = np.equal(r.v, [[2, 2, 2], [1, 2, 1], [0, 1, 1]])
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testSubtract(self):
        vecs = NumpyVectors([[2, 2, 2], [1, 2, 2], [2, 2, 3]])
        r = vecs - 1
        r = np.equal(r.v, [[1, 1, 1], [0, 1, 1], [1, 1, 2]])
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testDivide(self):
        vecs = NumpyVectors([[1, 1, 1], [-2, -2, -2], [3, 3, 3], [0, 0, 0]])
        r = vecs/2.0
        r = np.equal(r.v, [[0.5, 0.5, 0.5], [-1, -1, -1], [1.5, 1.5, 1.5], [0, 0, 0]])
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testMul(self):
        vecs = NumpyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = vecs * 2.0
        r = np.equal(r.v, [[-2, -2, -2], [4, 4, 4], [0, 0, 0]])
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testNeg(self):
        pass

    def testLen(self):
        vecs = NumpyVectors([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        r = len(vecs)
        self.assertEqual(r, 4)

    def testIsNullTrue(self):
        vecs = NumpyVectors([[0, 0, 0], [0, 0, 0]])
        r = vecs.isNull
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testIsNullFalse(self):
        vecs = NumpyVectors([[0, 0, 0], [0.0001, 0, 0]])
        r = vecs.isNull
        if False in r:
            r = False
        else:
            r = True
        self.assertFalse(r)

    def testIsUnitaryTrue(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        verify = np.count_nonzero(vecs.isUnitary)
        self.assertEqual(verify, 3)

    def testIsUnitaryFalse(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        verify = 4 - np.count_nonzero(vecs.isUnitary)
        self.assertEqual(verify, 1)

    def testRandomVectors(self):
        vecs = NumpyVectors.randomUniform(3, 3)
        print(vecs)
        r = np.less_equal(np.subtract(np.linalg.norm(vecs.v, axis=1), (np.ones((3, 3))*3).astype('float64')), 1e-9)
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testRandomUnitVectors(self):
        vecs = NumpyVectors.randomUniformUnitary(3)
        print(vecs.v[0, 0], vecs.v[0, 1], vecs.v[0, 2])
        r = vecs.isUnitary

        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testIsEqualToFalse(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        vecs2 = NumpyVectors([[0, 1, 1], [-0.0429843, 0.99337274, -0.10659786], [1, 1, 3]])
        r = vecs.isEqualTo(vecs2)
        if False in r:
            r = False
        else:
            r = True
        self.assertFalse(r)

    def testIsEqualToTrue(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        vecs2 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        r = vecs.isEqualTo(vecs2)
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testIsAlmostEqualToFalse(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        vecs2 = NumpyVectors([[1, 1, 1], [-0.04288243, 0.99337274, -0.10659786], [1, 1, 3]])
        r = vecs.isAlmostEqualTo(vecs2, 0.000000001)
        if False in r:
            r = False
        else:
            r = True
        self.assertFalse(r)

    def testIsAlmostEqualToTrue(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        vecs2 = NumpyVectors([[1, 1, 1], [-0.04298343, 0.99337274, -0.10659786], [1, 1, 3]])

        r = vecs.isAlmostEqualTo(vecs2, 0.00001)
        print(r.v)

        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testNormOutputTypeFloat64(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        vnorms = vecs.norm()
        print(vnorms[0])
        self.assertEqual(np.float64, type(vnorms[0]))

    def testNorm(self):
        vecs = NumpyVectors([[1, 1, 1], [0.866539324968574, -0.49677441419390916, 0.04821596919389434], [1, 1, 3]])
        vnorms = vecs.norm()
        print(vnorms.v[0], vnorms.v[1], vnorms.v[2])
        norm1 = np.sqrt(1 + 1 + 1)
        norm2 = 1
        norm3 = np.sqrt(1 + 1 + 3**2)
        norms = [norm1, norm2, norm3]
        print(norms)
        r = vnorms.isEqualTo(norms)

        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testAbs(self):
        vecs = NumpyVectors([[-1, 1, 1], [0.866539324968574, -0.49677441419390916, 0.04821596919389434], [1, 1, 3]])
        r = vecs.abs()
        r = np.greater_equal(r.v, np.zeros((3, 3)))
        if False in r:
            r = False
        else:
            r = True
        self.assertTrue(r)

    def testIsParallelToTrue(self):
        pass

    def testIsPerpendicularTo(self):
        pass

    def testDot(self):
        vecs3 = NumpyVectors([[0, 0, 1], [1, 1, 1], [-2, -2, -2]])
        vecs4 = NumpyVectors([[1, 0, 0], [1, 1, 1], [2, 2, 2]])
        r = vecs3.dot(vecs4)
        r = np.all(np.equal([0, 3, -12], r.v))
        self.assertTrue(r)

    def testCross(self):
        pass

    def testAnyPerpendicular(self):
        pass

    def testAnyUnitaryPerpendicular(self):
        pass

    def testIsInXYPlane(self):
        pass

    def testIsInYZPlane(self):
        pass

    def testIsInZXPlane(self):
        pass

    def testIsInPlane(self):
        pass

    def testNormalize(self):
        pass

    def testNormalized(self):
        pass

    def testNormalizedCrossProduct(self):
        pass

    def testNormalizedDotProduct(self):
        vecs = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        vecs2 = NumpyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0]])
        r = vecs.normalizedDotProduct(vecs2)
        r = np.equal(vecs.v, )


    def testAngleWith(self):
        pass

    def testPlaneOfIncidence(self):
        pass

    def testAngleOfIncidence(self):
        pass

    def testRotateAround(self):
        pass


if __name__ == '__main__':
    envtest.main()
