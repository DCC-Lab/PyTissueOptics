import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")


class TestVectorsBase(envtest.PyTissueTestCase):
    def setUp(self):
        self.count = 1000
        self.vnull = Vectors(N=self.count)
        self.vRandom = Vectors(self.randomVectors(N=1000))

    def tearDown(self):
        pass

    def testNullVector(self):
        self.assertIsNotNone(self.vnull)
        self.assertEqual(self.vnull.count, self.count)

        for v in self.vnull:        
            self.assertEqual(v.x, 0)
            self.assertEqual(v.y, 0)
            self.assertEqual(v.z, 0)

    def testSomeVectors(self):
        self.assertIsNotNone(self.vRandom)
        self.assertEqual(self.vRandom.count, 1000)

        for v in self.vRandom:        
            self.assertTrue(abs(v.x) <= 1)
            self.assertTrue(abs(v.y) <= 1)
            self.assertTrue(abs(v.z) <= 1)

    def testSomeVectors(self):
        self.assertEqual(len(self.vRandom), self.vRandom.count)

    def testReplaceSomeVectors(self):
        vs = self.vRandom
        vr = self.vnull

        vs.select([False]*1000)
        vs.selected[0] = True
        vs.replaceSelected(vr)
        self.assertEqual(vs[0], vr[0])
        self.assertNotEqual(vs[1], vr[1])

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
        self.assertEqual(v1.isParallelTo(v2, epsilon=1e-6), [True, True, False])

    def testVectorPerpendicular2(self):
        v1 = Vectors([zHat, xHat, yHat])
        v2 = Vectors([Vector(1, 1, 1), zHat, xHat])
        self.assertEqual(v1.isPerpendicularTo(v2, epsilon=1e-6), [False, True, True])

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
            self.assertAlmostEqual(angles[i], actual[i])

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
            self.assertAlmostEqual(angle[i], actual[i], 3, "Error at {0}".format(i))

    def testRotatedAround(self):
        v1 = Vectors([xHat, yHat, zHat])
        axis = Vectors([zHat, xHat, yHat])
        angle = [np.pi/2, np.pi/2, np.pi/2]

        v3 = v1.rotatedAround(axis, angle)

        actual = v1.angleWith(v3, axis)

        for i in range(3):
            self.assertAlmostEqual(angle[i], actual[i], 3, "Error at {0}".format(i))

    def testPlaneAndAngleOfIncidence(self):
        v1 = Vectors([Vector(0,1,1), Vector(0, -1, 1), Vector(1, 0, 1)])
        normal = Vectors([zHat, zHat, -zHat])

        expectedAngles = [np.pi/4, np.pi/4, np.pi/4]
        expectedPlanes = Vectors([xHat, -xHat, -yHat])
        actualAngles, actualPlanes, correctedNormal = v1.angleOfIncidence(normal)
        print(actualPlanes.v)
        for i in range(3):
            self.assertTrue(actualPlanes[i].isAlmostEqualTo(expectedPlanes[i]))
            self.assertAlmostEqual(actualAngles[i], expectedAngles[i])

    def testNormAbs(self):
        v1 = Vectors([Vector(1, 1, 1), Vector(1, 0, 1),Vector(0, 1, 1)])
        self.assertEqual(v1.norm(), [3, 2, 2])
        self.assertEqual(v1.abs(), [np.sqrt(3), np.sqrt(2), np.sqrt(2)])

    def testNormalize(self):
        v1 = Vectors([Vector(1, 1, 1), Vector(1, 0, 1),Vector(0, 1, 1)])
        v2 = v1.normalized()
        self.assertTrue(np.array(v2.isUnitary).all())

        v1.normalize()
        self.assertTrue(np.array(v1.isUnitary).all())

    def testNormalizeWithMask(self):
        v1 = Vectors([Vector(1, 1, 1), Vector(1, 0, 1),Vector(0, 1, 1)])
        v1.selected = [False, True, True]
        v2 = v1.normalized()

        v1.normalize()
        self.assertEqual(v1[0], Vector(1, 1, 1))

    def testScaledSum(self):
        v1 = Vectors([Vector(1, 2, 3), Vector(4, 5, 6),Vector(7, 8, 9)])
        v1.selected = [False, True, True]
        v2 = Vectors([Vector(1, 2, 3), Vector(4, 5, 6),Vector(7, 8, 9)])
        s = [1, 2, 3]
        v3 = Vectors([Vector(1, 2, 3), Vector(12, 15, 18), Vector(28, 32, 36)])
        r = Vectors.fromScaledSum(v1, v2, s)
        self.assertEqual(r, v3)
        self.assertEqual(v1.addScaled(v2, s), v3)

    def testAnyPerpendicular(self):
        vectors = Vectors([Vector(1, 2, 3), Vector(-1, -2, -3), xHat, yHat, zHat])
        for v in vectors:
            self.assertTrue(v.anyPerpendicular().isPerpendicularTo(v))

        self.assertIsNone(oHat.anyPerpendicular())
        self.assertIsNone(oHat.anyPerpendicular())

    def testAnyPerpendicularRandom(self):
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

    def randomVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2-1
        return Vector(x, y, z)

    def randomAngles(self, N):
        scalar = []
        for i in range(N):
            scalar.append((random.random() - 0.5) * 2 * np.pi)
        return scalar

    def randomVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random() * 2 - 1
            y = random.random() * 2 - 1
            z = random.random() * 2 - 1
            vectors.append(Vector(x, y, z))
        return vectors

    def randomUnitVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append(UnitVector(x, y, z))
        return vectors


class TestNumpyVectors(envtest.PyTissueTestCase):

    def testInitWithList(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        r = np.all(np.equal(v1.v, [[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]]))
        self.assertTrue(r)

    def testInitWithNumpyArray(self):
        v1 = NumpyVectors(np.array([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]]))
        r = np.all(np.equal(v1.v, [[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]]))
        self.assertTrue(r)

    def testCheckInitTypeFloat64(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        self.assertEqual(np.float64, type(v1[0][0]))

    def testGetItem(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        r = v1[0]
        r = np.all(np.equal(r, [1, 1, 1]))
        self.assertTrue(r)

    def testAdd(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 1, 0], [-1, 0, 0]])
        r = v1 + 1
        r = np.all(np.equal(r.v, [[2, 2, 2], [1, 2, 1], [0, 1, 1]]))
        self.assertTrue(r)

    def testSubtract(self):
        v1 = NumpyVectors([[2, 2, 2], [1, 2, 2], [2, 2, 3]])
        r = v1 - 1
        r = np.all(np.equal(r.v, [[1, 1, 1], [0, 1, 1], [1, 1, 2]]))
        self.assertTrue(r)

    def testDivide(self):
        v1 = NumpyVectors([[1, 1, 1], [-2, -2, -2], [3, 3, 3], [0, 0, 0]])
        r = v1/2.0
        r = np.all(np.equal(r.v, [[0.5, 0.5, 0.5], [-1, -1, -1], [1.5, 1.5, 1.5], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulFloat(self):
        v1 = NumpyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * 2.0
        r = np.all(np.equal(r.v, [[-2, -2, -2], [4, 4, 4], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulScalars(self):
        v1 = NumpyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * NumpyScalars([2, 1, 1])
        r = np.all(np.equal(r.v, [[-2, -2, -2], [2, 2, 2], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulVectorsUnique(self):
        v1 = NumpyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * NumpyVectors([2, 1, 1])
        r = np.all(np.equal(r.v, [[-2, -1, -1], [4, 2, 2], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulVectorsMultiple(self):
        v1 = NumpyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * NumpyVectors([[2, 1, 1], [0, 0, 1], [1, 1, 1]])
        r = np.all(np.equal(r.v, [[-2, -1, -1], [0, 0, 2], [0, 0, 0]]))
        self.assertTrue(r)

    def testNeg(self):
        v1 = NumpyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        v1 = -v1
        r = np.all(np.equal(v1.v, [[1, 1, 1], [-2, -2, -2], [0, 0, 0]]))
        self.assertTrue(r)

    def testLen(self):
        v1 = NumpyVectors([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        r = len(v1)
        self.assertEqual(r, 4)

    def testIsNullTrue(self):
        v1 = NumpyVectors([[0, 0, 0], [0, 0, 0]])
        r = np.all(v1.isNull)
        self.assertTrue(r)

    def testIsNullFalse(self):
        v1 = NumpyVectors([[0, 0, 0], [0.0001, 0, 0]])
        r = np.all(v1.isNull)
        self.assertFalse(r)

    def testIsUnitaryTrue(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        verify = np.count_nonzero(v1.isUnitary)
        self.assertEqual(verify, 3)

    def testIsUnitaryFalse(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        verify = 4 - np.count_nonzero(v1.isUnitary)
        self.assertEqual(verify, 1)

    def testRandomVectors(self):
        v1 = NumpyVectors.randomUniform(3, 3)
        r = np.all(np.less_equal(np.subtract(np.linalg.norm(v1.v, axis=1), (np.ones((3, 3))*3).astype('float64')), 1e-9))
        self.assertTrue(r)

    def testRandomUnitVectors(self):
        v1 = NumpyVectors.randomUniformUnitary(3)
        r = np.all(v1.isUnitary)
        self.assertTrue(r)

    def testIsEqualToFalse(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = NumpyVectors([[0, 1, 1], [-0.0429843, 0.99337274, -0.10659786], [1, 1, 3]])
        r = np.all(v1.isEqualTo(v2))
        self.assertFalse(r)

    def testIsEqualToTrue(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        r = np.all(v1.isEqualTo(v2))
        self.assertTrue(r)

    def testIsAlmostEqualToFalse(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = NumpyVectors([[1, 1, 1], [-0.04288243, 0.99337274, -0.10659786], [1, 1, 3]])
        r = np.all(v1.isAlmostEqualTo(v2, 0.000000001).v)
        self.assertFalse(r)

    def testIsAlmostEqualToTrue(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = NumpyVectors([[1, 1, 1], [-0.04298343, 0.99337274, -0.10659786], [1, 1, 3]])
        r = np.all(v1.isAlmostEqualTo(v2, 0.00001))
        self.assertTrue(r)

    def testNormOutputTypeFloat64(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v1norms = v1.norm()
        self.assertEqual(np.float64, type(v1norms[0]))

    def testNorm(self):
        v1 = NumpyVectors([[1, 1, 1], [0.866539324968574, -0.49677441419390916, 0.04821596919389434], [1, 1, 3]])
        v1norms = v1.norm()
        norm1 = np.sqrt(1 + 1 + 1)
        norm2 = 1
        norm3 = np.sqrt(1 + 1 + 3**2)
        norms = [norm1, norm2, norm3]
        r = np.all(np.isclose(norms, v1norms.v, atol=1e-7))
        self.assertTrue(r)

    def testNormNegative(self):
        v1 = NumpyVectors([[-1, -1, -1]])
        v1norms = v1.norm()
        r = np.all(np.isclose([np.sqrt(3)], v1norms.v, atol=1e-7))
        self.assertTrue(r)

    def testNormNull(self):
        v1 = NumpyVectors([[0, 0, 0]])
        v1norms = v1.norm()
        r = np.all(np.isclose([0], v1norms.v, atol=1e-7))
        self.assertTrue(r)

    def testAbs(self):
        v1 = NumpyVectors([[-1, 1, 1], [0.866539324968574, -0.49677441419390916, 0.04821596919389434], [1, 1, 3]])
        r = v1.abs()
        r = np.all(np.greater_equal(r.v, np.zeros((3, 3))))
        self.assertTrue(r)

    def testIsParallelToTrue(self):
        v1 = NumpyVectors([[2, 2, 2], [1, 1, 1], [-1, 1, 1]])
        v2 = NumpyVectors([[2, 2, 2], [-1, -1, -1], [-2, 2, 2]])
        r = v1.isParallelTo(v2)
        r = np.all(np.equal([True, True, True], r))
        self.assertTrue(r)

    def testIsParallelToFalse(self):
        v1 = NumpyVectors([[-1, 1, 1], [1, 1, 1], [-2, -2, -2]])
        v2 = NumpyVectors([[2, 2, 2], [-0.04, 1, -1], [1, 0.001, 0.001]])
        r = v1.isParallelTo(v2)
        r = np.all(np.equal([False, False, False], r))
        self.assertTrue(r)

    def testIsParallelToLimit(self):
        v1 = NumpyVectors([[1, 1, 1], [-1, -1, -1]])
        v2 = NumpyVectors([[0.99999, 0.99999, 0.99999], [-0.999, -1.0001, -1]])
        r = v1.isParallelTo(v2, epsilon=1e-5)
        r = np.all(np.equal([True, False], r))
        self.assertTrue(r)

    def testIsParallelToNull(self):
        v1 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        v2 = NumpyVectors([[2, 2, 2], [0, 0, 0]])
        r = v1.isParallelTo(v2)
        r = np.all(np.equal([False, False], r))
        self.assertTrue(r)

    def testIsPerpendicularToTrue(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 1], [-1, 0, 0]])
        v2 = NumpyVectors([[-2, 1, 1], [1, 0, 0], [0, 1, 0]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([True, True, True], r))
        self.assertTrue(r)

    def testIsPerpendicularToFalse(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 1], [-1, 0, 0]])
        v2 = NumpyVectors([[-1.999, 1, 1], [0.999, 0, 0.1], [0.1, 0.999, 0]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([False, False, False], r))
        self.assertTrue(r)

    def testIsPerpendicularToLimit(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 1], [-1, 0, 0]])
        v2 = NumpyVectors([[-1.999999999, 1, 1], [0.000000001, 1, 0.000000001], [0.000000001, 1, 0]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([True, True, True], r))
        self.assertTrue(r)

    def testIsPerpendicularToNull(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([False, False], r))
        self.assertTrue(r)

    def testDot(self):
        v1 = NumpyVectors([[0, 0, 1], [1, 1, 1], [-2, -2, -2]])
        v2 = NumpyVectors([[1, 0, 0], [1, 1, 1], [2, 2, 2]])
        r = v1.dot(v2)
        r = np.all(np.equal([0, 3, -12], r.v))
        self.assertTrue(r)

    def testDotNull(self):
        v1 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        v2 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
        r = v1.dot(v2)
        r = np.all(np.equal([0, 0], r.v))
        self.assertTrue(r)

    def testCross(self):
        v1 = NumpyVectors([[0, 0, 1], [1, 1, 1], [-2, -2, -2]])
        v2 = NumpyVectors([[1, 0, 0], [-1, -1, -1], [1, 0, 0]])
        r = v1.cross(v2)
        r = np.all(np.equal([[0, 1, 0], [0, 0, 0], [0, -2, 2]], r.v))
        self.assertTrue(r)

    def testCrossNull(self):
        v1 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        v2 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
        r = v1.cross(v2)
        r = np.all(np.equal([[0, 0, 0], [0, 0, 0]], r.v))
        self.assertTrue(r)

    def testAnyPerpendicularMasks(self):
        vecs = NumpyVectors([[0, 0, 1], [1, 1, 1], [2, 2, 0], [0, 0, 0], [1, 0, 0]])
        convA = np.array([1, 1, 0]).transpose()
        convB = np.array([0, 0, 1]).transpose()
        YZ0 = vecs * convA
        Z0 = vecs * convB
        maskXY0 = np.all(YZ0.v == 0, axis=1)
        maskZ0 = np.all(Z0.v == 0, axis=1)
        maskXYZ0 = np.logical_and(maskXY0, maskZ0)
        maskXYZ1 = np.invert(np.logical_or(maskXYZ0, maskZ0))
        verify = [[True,False,False,True,False],[False,False,True,True,True],[False,False,False,True,False],[True,True,False,False,False]]
        r = np.all([np.equal(verify[0], maskXY0), np.equal(verify[1], maskZ0), np.equal(verify[2], maskXYZ0), np.equal(verify[3], maskXYZ1)])
        self.assertTrue(r)

    def testAnyPerpendicular(self):
        v1 = NumpyVectors([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [-1, -2, -3], [1, 1, 1], [-1, -1, -1]])
        r = v1.anyPerpendicular()
        r = np.all(r.isPerpendicularTo(v1))
        self.assertTrue(r)

    def testAnyPerpendicularNull(self):
        v1 = NumpyVectors([[0, 0, 0]])
        r = v1.anyPerpendicular()
        self.assertTrue(np.all(np.isnan(r.v)))

    def testAnyUnitaryPerpendicular(self):
        v1 = NumpyVectors([[0, 0, 1], [1, 1, 1], [2, 2, 0], [1, 0, 0]])
        r = v1.anyUnitaryPerpendicular()
        r = np.all(r.isUnitary)
        self.assertTrue(r)

    def testAnyUnitaryPerpendicularNull(self):
        v1 = NumpyVectors([[0, 0, 0]])
        r = v1.anyUnitaryPerpendicular()
        r = np.all(np.isnan(r.v))
        self.assertTrue(r)

    def testIsInXYPlane(self):
        pass

    def testIsInYZPlane(self):
        pass

    def testIsInZXPlane(self):
        pass

    def testIsInPlane(self):
        pass

    def testNormalize(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        norm1 = v1.norm().v
        verify = np.where(norm1 != 0, 1, 0)
        v1.normalize()
        norm2 = v1.norm().v
        r = np.all(np.equal(norm2, verify))
        self.assertTrue(r)

    def testNormalizeNullVector(self):
        with self.assertRaises(ValueError):
            v1 = NumpyVectors([[0, 0, 0]])
            v1.normalize()

    def testNormalizedIndependentObject(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0]])
        v2 = v1.normalized()
        v2Norm = v2.norm().v
        v1Norm = v1.norm().v
        verify = np.isclose(v1Norm, v2Norm, atol=1e-8)
        r = np.all(np.equal([0, 1, 0, 1], verify))
        self.assertTrue(r)

    def testNormalizedCrossProduct(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0], [0, 0, 0]])
        v2 = NumpyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0], [1, 1, 1]])
        r = v1.normalizedCrossProduct(v2)
        self.assertTrue(np.all(np.isclose([[-0.4082482905, 0.4082482905, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]], r.v, atol=1e-9)))

    def testNormalizedCrossProductNullVectors(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        r = v1.normalizedCrossProduct(v2)
        self.assertTrue(np.all(np.isclose([[0, 0, 0], [0, 0, 0]], r.v, atol=1e-9)))

    def testNormalizedDotProduct(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        v2 = NumpyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0]])
        r = v1.normalizedDotProduct(v2)
        self.assertTrue(np.all(np.isclose([0.8164965809, 1, 0, 0], r.v, atol=1e-9)))

    def testNormalizedDotProductNullVectors(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        r = v1.normalizedDotProduct(v2)
        self.assertTrue(np.all(np.isclose([0, 0], r.v, atol=1e-9)))

    def testAngleWith(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0], [0, 0, 0]])
        v2 = NumpyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0], [1, 1, 1]])
        axis = NumpyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.angleWith(v2, axis)
        # le dernier -3.14 is the same for Vector, but seems an error, cause it should probably be 0.
        self.assertTrue(np.all(np.isclose([0.61547971, 0, -1.57079633, -1.57079633, -3.14159265], r.v, atol=1e-7)))

    def testAngleWithNullVectors(self):
        v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = NumpyVectors([[0, 0, 0], [1, 1, 1]])
        axis = NumpyVectors([[1, 1, 1], [1, 1, 1]])
        r = v1.angleWith(v2, axis)
        # Check that angle is very close to <(1e-7) 0 or +-pi
        self.assertTrue(np.all(np.logical_or(np.isclose([3.14159265, 3.14159265], np.abs(r.v), atol=1e-7), np.isclose([0, 0], r.v, atol=1e-7))))

    def testPlaneOfIncidence(self):
        v1 = NumpyVectors([[1, 1,  1], [-0.04298243, 0.99337274, -0.10659786], [-1, 1, 0]])
        normal = NumpyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.planeOfIncidence(normal)

        self.assertTrue(np.all(np.isclose(
            [[-0.70710678, 0, 0.70710678],
             [0.92744322,  0, -0.37396401],
             [0,           0,          -1]], r.v, atol=1e-7)))

    def testPlaneOfIncidenceIsUnitary(self):
        v1 = NumpyVectors([[1, 1,  1], [-0.04298243, 0.99337274, -0.10659786], [-1, 1, 0]])
        normal = NumpyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.planeOfIncidence(normal)
        self.assertTrue(np.all(r.isUnitary))

    def testPlaneOfIncidenceParallelToNormal(self):
        v1 = NumpyVectors([[1, 1, 1], [-1, -1, -1], [-1, -1, -1], [1, 1, 1]])
        normal = NumpyVectors([[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1]])
        r = v1.planeOfIncidence(normal)
        self.assertTrue(np.all(np.isclose(
            [[0, -0.70710678,  0.70710678],
             [0,  0.70710678, -0.70710678],
             [0,  0.70710678, -0.70710678],
             [0, -0.70710678,  0.70710678]], r.v, atol=1e-7)))

    def testPlaneOfIncidencePerpendicularToNormal(self):
        v1 = NumpyVectors([[1, 0, 0]])
        normal = NumpyVectors([[0, 1, 0]])
        r = v1.planeOfIncidence(normal)
        self.assertTrue(np.all(np.isclose([[0, 0, 1]], r.v, atol=1e-7)))

    def testPlaneOfIncidenceNullVectors(self):
        with self.assertRaises(ValueError):
            v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
            normal = NumpyVectors([[0, 0, 0], [1, 1, 1]])
            r = v1.planeOfIncidence(normal)

    def testAngleOfIncidence(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0]])
        plane = NumpyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.angleOfIncidence(plane)
        self.assertTrue(np.all(np.isclose([0.95531662, 0.11519193, 0, 1.57079633], r[0].v, atol=1e-7)))

    def testAngleOfIncidenceParallelToNormal(self):
        v1 = NumpyVectors([[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1]])
        plane = NumpyVectors([[1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1]])
        r = v1.angleOfIncidence(plane)
        self.assertTrue(np.all(np.isclose([0, 0, 0, 0], r[0].v, atol=1e-7)))

    def testAngleOfIncidencePerpendicularToNormal(self):
        v1 = NumpyVectors([[1, 0, 0]])
        plane = NumpyVectors([[0, 1, 0]])
        r = v1.angleOfIncidence(plane)
        self.assertTrue(np.all(np.isclose([1.57079632679], np.abs(r[0].v), atol=1e-7)))

    def testAngleOfIncidenceNull(self):
        with self.assertRaises(ValueError):
            v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
            plane = NumpyVectors([[0, 0, 0], [1, 1, 1]])
            v1.angleOfIncidence(plane)

    def testRotateAround(self):
        v1 = NumpyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0]])
        axis = NumpyVectors([[0, 1, 0], [0, -1, 0], [1, 1, 0], [0, 1, 0]])
        theta = NumpyScalars([3.1415, 1.618, 0.1, 1])
        r = v1.rotateAround(axis, theta)
        self.assertTrue(np.all(np.isclose(
            [[-0.99990734,  1,         -1.00009265],
             [0.1085073,   0.99337274, -0.03790461],
             [0.00499583,  1.99500417,  0.14118577],
             [-0.54030231,  0,          0.84147098]], r.v, atol=1e-7)))

    def testRotateAroundParallel(self):
        v1 = NumpyVectors([[1, 1, 1]])
        axis = NumpyVectors([[1, 1, 1]])
        theta = NumpyScalars([3.14159265])
        r = v1.rotateAround(axis, theta)
        self.assertTrue(np.all(np.isclose(
            [[1, 1, 1]], r.v, atol=1e-7)))

    def testRotateAroundNullVectors(self):
        with self.assertRaises(ValueError):
            v1 = NumpyVectors([[1, 1, 1], [0, 0, 0]])
            axis = NumpyVectors([[0, 0, 0], [1, 1, 1]])
            theta = NumpyScalars([1, 1])
            v1.rotateAround(axis, theta)


class TestCupyVectors(envtest.PyTissueTestCase):

    def testInitWithList(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        r = cp.all(cp.equal(v1.v, cp.asarray([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])))
        self.assertTrue(r)

    def testInitWithNumpyArray(self):
        v1 = CupyVectors(cp.array([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]]))
        r = cp.all(cp.equal(v1.v, cp.asarray([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])))
        self.assertTrue(r)

    def testCheckInitTypeFloat64(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        self.assertEqual(cp.dtype("float64").type, v1.v[0, 0].dtype)

    def testGetItem(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        r = v1[0]
        r = np.all(np.equal(cp.asnumpy(r), [1, 1, 1]))
        self.assertTrue(r)

    def testAdd(self):
        v1 = CupyVectors([[1, 1, 1], [0, 1, 0], [-1, 0, 0]])
        r = v1 + 1
        r = np.all(np.equal(cp.asnumpy(r.v), [[2, 2, 2], [1, 2, 1], [0, 1, 1]]))
        self.assertTrue(r)

    def testSubtract(self):
        v1 = CupyVectors([[2, 2, 2], [1, 2, 2], [2, 2, 3]])
        r = v1 - 1
        r = np.all(np.equal(cp.asnumpy(r.v), [[1, 1, 1], [0, 1, 1], [1, 1, 2]]))
        self.assertTrue(r)

    def testDivide(self):
        v1 = CupyVectors([[1, 1, 1], [-2, -2, -2], [3, 3, 3], [0, 0, 0]])
        r = v1/2.0
        r = np.all(np.equal(cp.asnumpy(r.v), [[0.5, 0.5, 0.5], [-1, -1, -1], [1.5, 1.5, 1.5], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulFloat(self):
        v1 = CupyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * 2.0
        r = np.all(np.equal(cp.asnumpy(r.v), [[-2, -2, -2], [4, 4, 4], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulScalars(self):
        v1 = CupyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * CupyScalars([2, 1, 1])
        r = np.all(np.equal(cp.asnumpy(r.v), [[-2, -2, -2], [2, 2, 2], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulVectorsUnique(self):
        v1 = CupyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * CupyVectors([2, 1, 1])
        r = np.all(np.equal(cp.asnumpy(r.v), [[-2, -1, -1], [4, 2, 2], [0, 0, 0]]))
        self.assertTrue(r)

    def testMulVectorsMultiple(self):
        v1 = CupyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = v1 * CupyVectors([[2, 1, 1], [0, 0, 1], [1, 1, 1]])
        r = np.all(np.equal(cp.asnumpy(r.v), [[-2, -1, -1], [0, 0, 2], [0, 0, 0]]))
        self.assertTrue(r)

    def testNeg(self):
        v1 = CupyVectors([[-1, -1, -1], [2, 2, 2], [0, 0, 0]])
        r = -v1
        r = np.all(np.equal(cp.asnumpy(r.v), [[1, 1, 1], [-2, -2, -2], [0, 0, 0]]))
        self.assertTrue(r)

    def testLen(self):
        v1 = CupyVectors([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        r = len(v1)
        self.assertEqual(r, 4)

    def testIsNullTrue(self):
        v1 = CupyVectors([[0, 0, 0], [0, 0, 0]])
        r = np.all(v1.isNull)
        self.assertTrue(r)

    def testIsNullFalse(self):
        v1 = CupyVectors([[0, 0, 0], [0.0001, 0, 0]])
        r = np.all(v1.isNull)
        self.assertFalse(r)

    def testIsUnitaryTrue(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        verify = np.count_nonzero(v1.isUnitary)
        self.assertEqual(verify, 3)

    def testIsUnitaryFalse(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        verify = 4 - np.count_nonzero(v1.isUnitary)
        self.assertEqual(verify, 1)

    def testRandomVectors(self):
        v1 = CupyVectors.randomUniform(3, 3)
        r = np.all(np.less_equal(np.subtract(np.linalg.norm(cp.asnumpy(v1.v), axis=1), (np.ones((3, 3))*3).astype('float64')), 1e-9))
        self.assertTrue(r)

    def testRandomUnitVectors(self):
        v1 = CupyVectors.randomUniformUnitary(3)
        r = np.all(v1.isUnitary)
        self.assertTrue(r)

    def testIsEqualToFalse(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = CupyVectors([[0, 1, 1], [-0.0429843, 0.99337274, -0.10659786], [1, 1, 3]])
        r = cp.all(v1.isEqualTo(v2).v)
        self.assertFalse(r)

    def testIsEqualToTrue(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        r = cp.all(v1.isEqualTo(v2).v)
        self.assertTrue(r)

    def testIsAlmostEqualToFalse(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = CupyVectors([[1, 1, 1], [-0.04288243, 0.99337274, -0.10659786], [1, 1, 3]])
        r = cp.all(v1.isAlmostEqualTo(v2, 0.000000001).v)
        self.assertFalse(r)

    def testIsAlmostEqualToTrue(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v2 = CupyVectors([[1, 1, 1], [-0.04298343, 0.99337274, -0.10659786], [1, 1, 3]])
        r = cp.all(v1.isAlmostEqualTo(v2, 0.00001).v)
        self.assertTrue(r)

    def testNormOutputTypeFloat64(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [1, 1, 3]])
        v1norms = v1.norm()
        self.assertEqual(np.dtype("float64").type, v1norms[0].dtype)

    def testNorm(self):
        v1 = CupyVectors([[1, 1, 1], [0.866539324968574, -0.49677441419390916, 0.04821596919389434], [1, 1, 3]])
        v1norms = v1.norm()
        norm1 = np.sqrt(1 + 1 + 1)
        norm2 = 1
        norm3 = np.sqrt(1 + 1 + 3**2)
        norms = [norm1, norm2, norm3]
        r = np.all(np.isclose(norms, cp.asnumpy(v1norms.v), atol=1e-7))
        self.assertTrue(r)

    def testNormNegative(self):
        v1 = CupyVectors([[-1, -1, -1]])
        v1norms = v1.norm()
        r = np.all(np.isclose([np.sqrt(3)], cp.asnumpy(v1norms.v), atol=1e-7))
        self.assertTrue(r)

    def testNormNull(self):
        v1 = CupyVectors([[0, 0, 0]])
        v1norms = v1.norm()
        r = np.all(np.isclose([0], cp.asnumpy(v1norms.v), atol=1e-7))
        self.assertTrue(r)

    def testAbs(self):
        v1 = CupyVectors([[-1, 1, 1], [0.866539324968574, -0.49677441419390916, 0.04821596919389434], [1, 1, 3]])
        r = v1.abs()
        r = np.all(np.greater_equal(cp.asnumpy(r.v), np.zeros((3, 3))))
        self.assertTrue(r)

    def testIsParallelToTrue(self):
        v1 = CupyVectors([[2, 2, 2], [1, 1, 1], [-1, 1, 1]])
        v2 = CupyVectors([[2, 2, 2], [-1, -1, -1], [-2, 2, 2]])
        r = v1.isParallelTo(v2)
        r = np.all(np.equal([True, True, True], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsParallelToFalse(self):
        v1 = CupyVectors([[-1, 1, 1], [1, 1, 1], [-2, -2, -2]])
        v2 = CupyVectors([[2, 2, 2], [-0.04, 1, -1], [1, 0.001, 0.001]])
        r = v1.isParallelTo(v2)
        r = np.all(np.equal([False, False, False], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsParallelToLimit(self):
        v1 = CupyVectors([[1, 1, 1], [-1, -1, -1]])
        v2 = CupyVectors([[0.99999, 0.99999, 0.99999], [-0.999, -1.0001, -1]])
        r = v1.isParallelTo(v2, epsilon=1e-5)
        r = np.all(np.equal([True, False], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsParallelToNull(self):
        v1 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        v2 = CupyVectors([[2, 2, 2], [0, 0, 0]])
        r = v1.isParallelTo(v2)
        r = np.all(np.equal([False, False], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsPerpendicularToTrue(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 1], [-1, 0, 0]])
        v2 = CupyVectors([[-2, 1, 1], [1, 0, 0], [0, 1, 0]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([True, True, True], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsPerpendicularToFalse(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 1], [-1, 0, 0]])
        v2 = CupyVectors([[-1.999, 1, 1], [0.999, 0, 0.1], [0.1, 0.999, 0]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([False, False, False], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsPerpendicularToLimit(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 1], [-1, 0, 0]])
        v2 = CupyVectors([[-1.999999999, 1, 1], [0.000000001, 1, 0.000000001], [0.000000001, 1, 0]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([True, True, True], cp.asnumpy(r)))
        self.assertTrue(r)

    def testIsPerpendicularToNull(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        r = v1.isPerpendicularTo(v2)
        r = np.all(np.equal([False, False], cp.asnumpy(r)))
        self.assertTrue(r)

    def testDot(self):
        v1 = CupyVectors([[0, 0, 1], [1, 1, 1], [-2, -2, -2]])
        v2 = CupyVectors([[1, 0, 0], [1, 1, 1], [2, 2, 2]])
        r = v1.dot(v2)
        r = np.all(np.equal([0, 3, -12], cp.asnumpy(r.v)))
        self.assertTrue(r)

    def testDotNull(self):
        v1 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        v2 = CupyVectors([[1, 1, 1], [0, 0, 0]])
        r = v1.dot(v2)
        r = np.all(np.equal([0, 0], cp.asnumpy(r.v)))
        self.assertTrue(r)

    def testCross(self):
        v1 = CupyVectors([[0, 0, 1], [1, 1, 1], [-2, -2, -2]])
        v2 = CupyVectors([[1, 0, 0], [-1, -1, -1], [1, 0, 0]])
        r = v1.cross(v2)
        r = np.all(np.equal([[0, 1, 0], [0, 0, 0], [0, -2, 2]], cp.asnumpy(r.v)))
        self.assertTrue(r)

    def testCrossNull(self):
        v1 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        v2 = CupyVectors([[1, 1, 1], [0, 0, 0]])
        r = v1.cross(v2)
        r = np.all(np.equal([[0, 0, 0], [0, 0, 0]], cp.asnumpy(r.v)))
        self.assertTrue(r)

    def testAnyPerpendicular(self):
        v1 = CupyVectors([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, 3], [-1, -2, -3], [1, 1, 1], [-1, -1, -1]])
        r = v1.anyPerpendicular()
        self.assertTrue(np.all(cp.asnumpy(r.isPerpendicularTo(v1).v)))

    def testAnyPerpendicularNull(self):
        v1 = CupyVectors([[0, 0, 0]])
        r = v1.anyPerpendicular()
        self.assertTrue(np.all(np.isnan(cp.asnumpy(r.v))))

    def testAnyUnitaryPerpendicular(self):
        v1 = CupyVectors([[0, 0, 1], [1, 1, 1], [2, 2, 0], [1, 0, 0]])
        r = v1.anyUnitaryPerpendicular()
        r = np.all(r.isUnitary)
        self.assertTrue(r)

    def testAnyUnitaryPerpendicularNull(self):
        v1 = CupyVectors([[0, 0, 0]])
        r = v1.anyUnitaryPerpendicular()
        r = np.all(np.isnan(cp.asnumpy(r.v)))
        self.assertTrue(r)

    def testIsInXYPlane(self):
        pass

    def testIsInYZPlane(self):
        pass

    def testIsInZXPlane(self):
        pass

    def testIsInplane(self):
        pass

    def testNormalize(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        norm1 = v1.norm().v
        verify = np.where(norm1 != 0, 1, 0)
        v1.normalize()
        norm2 = v1.norm().v
        r = np.all(np.equal(norm2, verify))
        self.assertTrue(r)

    def testNormalizeNullVector(self):
        with self.assertRaises(ValueError):
            v1 = CupyVectors([[0, 0, 0]])
            v1.normalize()

    def testNormalizedIndependentObject(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0]])
        v2 = v1.normalized()
        v2Norm = v2.norm().v
        v1Norm = v1.norm().v
        verify = np.isclose(v1Norm, v2Norm, atol=1e-8)
        r = np.all(np.equal([0, 1, 0, 1], cp.asnumpy(verify)))
        self.assertTrue(r)

    def testNormalizedCrossProduct(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0], [0, 0, 0]])
        v2 = CupyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0], [1, 1, 1]])
        r = v1.normalizedCrossProduct(v2)
        self.assertTrue(np.all(np.isclose([[-0.4082482905, 0.4082482905, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]], r.v, atol=1e-9)))

    def testNormalizedCrossProductNullVectors(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        r = v1.normalizedCrossProduct(v2)
        self.assertTrue(np.all(np.isclose([[0, 0, 0], [0, 0, 0]], r.v, atol=1e-9)))

    def testNormalizedDotProduct(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 1, 0], [-1, 0, 0]])
        v2 = CupyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0]])
        r = v1.normalizedDotProduct(v2)
        self.assertTrue(np.all(np.isclose([0.8164965809, 1, 0, 0], r.v, atol=1e-9)))

    def testNormalizedDotProductNullVectors(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        r = v1.normalizedDotProduct(v2)
        self.assertTrue(np.all(np.isclose([0, 0], r.v, atol=1e-9)))

    def testAngleWith(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0], [0, 0, 0]])
        v2 = CupyVectors([[1, 1, 0], [-0.04298243, 0.99337274, -0.10659786], [0, 0, 1], [0, -2, 0], [1, 1, 1]])
        axis = CupyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.angleWith(v2, axis)
        # le dernier -3.14 is the same for Vector, but seems an error, cause it should probably be 0.
        self.assertTrue(np.all(np.isclose([0.61547971, 0, -1.57079633, -1.57079633, -3.14159265], r.v, atol=1e-7)))

    def testAngleWithNullVectors(self):
        v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
        v2 = CupyVectors([[0, 0, 0], [1, 1, 1]])
        axis = CupyVectors([[1, 1, 1], [1, 1, 1]])
        r = v1.angleWith(v2, axis)
        # Check that angle is very close to <(1e-7) 0 or +-pi
        self.assertTrue(np.all(np.logical_or(np.isclose([3.14159265, 3.14159265], np.abs(r.v), atol=1e-7), np.isclose([0, 0], r.v, atol=1e-7))))

    def testPlaneOfIncidence(self):
        v1 = CupyVectors([[1, 1,  1], [-0.04298243, 0.99337274, -0.10659786], [-1, 1, 0]])
        normal = CupyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.planeOfIncidence(normal)

        self.assertTrue(np.all(np.isclose(
            [[-0.70710678, 0, 0.70710678],
             [0.92744322,  0, -0.37396401],
             [0,           0,          -1]], r.v, atol=1e-7)))

    def testPlaneOfIncidenceIsUnitary(self):
        v1 = CupyVectors([[1, 1,  1], [-0.04298243, 0.99337274, -0.10659786], [-1, 1, 0]])
        normal = CupyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.planeOfIncidence(normal)
        self.assertTrue(np.all(r.isUnitary))

    def testPlaneOfIncidenceParallelToNormal(self):
        v1 = CupyVectors([[1, 1, 1], [-1, -1, -1], [-1, -1, -1], [1, 1, 1]])
        normal = CupyVectors([[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1]])
        r = v1.planeOfIncidence(normal)
        self.assertTrue(np.all(np.isclose(
            [[0, -0.70710678,  0.70710678],
             [0,  0.70710678, -0.70710678],
             [0,  0.70710678, -0.70710678],
             [0, -0.70710678,  0.70710678]], r.v, atol=1e-7)))

    def testPlaneOfIncidencePerpendicularToNormal(self):
        v1 = CupyVectors([[1, 0, 0]])
        normal = CupyVectors([[0, 1, 0]])
        r = v1.planeOfIncidence(normal)
        self.assertTrue(np.all(np.isclose([[0, 0, 1]], r.v, atol=1e-7)))

    def testPlaneOfIncidenceNullVectors(self):
        with self.assertRaises(ValueError):
            v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
            normal = CupyVectors([[0, 0, 0], [1, 1, 1]])
            r = v1.planeOfIncidence(normal)

    def testAngleOfIncidence(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0]])
        plane = CupyVectors([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        r = v1.angleOfIncidence(plane)
        self.assertTrue(np.all(np.isclose([0.95531662, 0.11519193, 0, 1.57079633], cp.asnumpy(r[0].v), atol=1e-7)))

    def testAngleOfIncidenceParallelToNormal(self):
        v1 = CupyVectors([[1, 1, 1], [1, 1, 1], [-1, -1, -1], [-1, -1, -1]])
        plane = CupyVectors([[1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, -1]])
        r = v1.angleOfIncidence(plane)
        self.assertTrue(np.all(np.isclose([0, 0, 0, 0], r[0].v, atol=1e-7)))

    def testAngleOfIncidencePerpendicularToNormal(self):
        v1 = CupyVectors([[1, 0, 0]])
        plane = CupyVectors([[0, 1, 0]])
        r = v1.angleOfIncidence(plane)
        self.assertTrue(np.all(np.isclose([1.57079632679], np.abs(r[0].v), atol=1e-7)))

    def testAngleOfIncidenceNull(self):
        with self.assertRaises(ValueError):
            v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
            plane = CupyVectors([[0, 0, 0], [1, 1, 1]])
            v1.angleOfIncidence(plane)

    def testRotateAround(self):
        v1 = CupyVectors([[1, 1, 1], [-0.04298243, 0.99337274, -0.10659786], [0, 2, 0], [-1, 0, 0]])
        axis = CupyVectors([[0, 1, 0], [0, -1, 0], [1, 1, 0], [0, 1, 0]])
        theta = NumpyScalars([3.1415, 1.618, 0.1, 1])
        r = v1.rotateAround(axis, theta)
        self.assertTrue(np.all(np.isclose(
            [[-0.99990734,  1,         -1.00009265],
             [0.1085073,   0.99337274, -0.03790461],
             [0.00499583,  1.99500417,  0.14118577],
             [-0.54030231,  0,          0.84147098]], r.v, atol=1e-7)))

    def testRotateAroundParallel(self):
        v1 = CupyVectors([[1, 1, 1]])
        axis = CupyVectors([[1, 1, 1]])
        theta = NumpyScalars([3.14159265])
        r = v1.rotateAround(axis, theta)
        self.assertTrue(np.all(np.isclose(
            [[1, 1, 1]], r.v, atol=1e-7)))

    def testRotateAroundNullVectors(self):
        with self.assertRaises(ValueError):
            v1 = CupyVectors([[1, 1, 1], [0, 0, 0]])
            axis = CupyVectors([[0, 0, 0], [1, 1, 1]])
            theta = NumpyScalars([1, 1])
            v1.rotateAround(axis, theta)


if __name__ == '__main__':
    envtest.main()
