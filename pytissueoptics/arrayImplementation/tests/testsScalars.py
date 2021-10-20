import unittest
from arrayImplementation.scalars import *
from arrayImplementation.vectors import *


class TestScalars(unittest.TestCase):

    def testScalarsCreation(self):
        with self.subTest("NumpyScalar"):
            with self.subTest("NotNull"):
                s = NumpyScalars(N=1000)
                self.assertIsNotNone(s)
                self.assertTrue(np.all(np.equal(s, 0)))

                s = NumpyScalars([0]*1000)
                self.assertIsNotNone(s)
                self.assertTrue(np.all(np.equal(s, 0)))

            with self.subTest("1 dimensional, proper lenght"):
                s = NumpyScalars(N=1000)
                self.assertEqual(len(s), 1000)

                s = NumpyScalars([0] * 1000)
                self.assertEqual(len(s), 1000)

            with self.subTest("variable type np.float64"):
                s = NumpyScalars(N=1000)
                self.assertEqual(s.v.dtype, "float64")

                s = NumpyScalars([0] * 1000)
                self.assertEqual(s.v.dtype, "float64")

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsAdd(self):
        with self.subTest("NumpyScalar"):
            with self.subTest("with Scalars"):
                s1 = NumpyScalars([1]*1000)
                s2 = NumpyScalars([2]*1000)
                sf = s1 + s2
                self.assertTrue(np.all(np.equal(sf, 3)))

                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyScalars([3, 4, 5])
                sf = s1 + s2
                self.assertTrue(np.all(np.equal(sf, [4, 6, 8])))

            with self.subTest("with Vectors"):
                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
                sf = s1 - s2
                self.assertEqual(sf, ArithmeticError)

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsSubtract(self):
        with self.subTest("NumpyScalar"):
            with self.subTest("with Scalars"):
                s1 = NumpyScalars([1]*1000)
                s2 = NumpyScalars([2]*1000)
                sf = s1 - s2
                self.assertTrue(np.all(np.equal(sf, -1)))

                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyScalars([3, 4, 3])
                sf = s1 - s2
                self.assertTrue(np.all(np.equal(sf, [-2, -2, 0])))

                s1 = NumpyScalars([1, 2, 3])
                s2 = 1
                sf = s1 - s2
                self.assertTrue(np.all(np.equal(sf, [0, 1, 2])))

            with self.subTest("with Vectors"):
                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
                sf = s1 - s2
                self.assertEqual(sf, ArithmeticError)

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsMultiply(self):
        with self.subTest("NumpyScalar"):
            with self.subTest("with Scalars"):
                s1 = NumpyScalars([2]*1000)
                s2 = NumpyScalars([2]*1000)
                sf = s1 * s2
                self.assertTrue(np.all(np.equal(sf, 4)))

                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyScalars([3, 4, 5])
                sf = s1 * s2
                self.assertTrue(np.all(np.equal(sf, [3, 8, 15])))

                s1 = NumpyScalars([1, 2, 3])
                s2 = 2
                sf = s1 * s2
                self.assertTrue(np.all(np.equal(sf, [2, 4, 6])))

            with self.subTest("with Vectors"):
                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
                sf = s1 * s2
                self.assertTrue(isinstance(sf, NumpyVectors))
                self.assertTrue(np.all(np.equal(sf.v, [[1, 1, 1], [4, 4, 4], [9, 9, 9]])))

                sf = s2*s1
                self.assertEqual(sf, ArithmeticError)

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsTrueDiv(self):
        with self.subTest("NumpyScalar"):
            with self.subTest("with Scalars"):
                s1 = NumpyScalars([2]*1000)
                s2 = NumpyScalars([2]*1000)
                sf = s1 / s2
                self.assertTrue(np.all(np.equal(sf, 1)))

                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyScalars([3, 4, 5])
                sf = s1 / s2
                self.assertTrue(np.all(np.equal(sf, [1/3, 2/4, 3/5])))

                s1 = NumpyScalars([1, 2, 3])
                s2 = 2
                sf = s1 / s2
                self.assertTrue(np.all(np.equal(sf, [1/2, 1, 3/2])))

            with self.subTest("with Vectors"):
                s1 = NumpyScalars([1, 2, 3])
                s2 = NumpyVectors([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
                sf = s1 / s2
                self.assertEqual(sf, ArithmeticError)

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsNeg(self):
        with self.subTest("NumpyScalar"):
            with self.subTest("with Scalars"):
                s1 = NumpyScalars([2] * 1000)
                s2 = NumpyScalars([-2] * 1000)
                s1 = -s1
                self.assertTrue(np.all(np.equal(s1, s2)))

                s1 = NumpyScalars([1] * 1000)
                s2 = NumpyScalars([0] * 1000)
                s1 = -s1
                self.assertFalse(np.all(np.equal(s1, s2)))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsInvert(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([0]*1000)
            self.assertFalse(s.all())
            s = ~s
            self.assertTrue(s.all())

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsGetItem(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s[3]
            self.assertTrue(np.all(np.equal(g.v, 4)))

            s = NumpyScalars([1, 2, 3, 4])
            #print(s)
            g = s[1:5]
            #print(g)
            self.assertTrue(np.all(np.equal(g.v, [2, 3, 4])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsSetItem(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            s[3] = 0
            self.assertTrue(np.all(np.equal(s.v, [1,2,3,0])))

            s = NumpyScalars([1, 2, 3, 4])
            s[0:2] = [0, 0]
            self.assertTrue(np.all(np.equal(s.v, [0, 0, 3, 4])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsAll(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1]*1000)
            self.assertTrue(s.all())

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsAny(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([0]*1000)
            self.assertFalse(s.any())

            s[0] = 1
            self.assertTrue(s.any())

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsConditional_le(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s.conditional_le(3, 0, s)
            self.assertTrue(np.all(np.equal(g.v, [0,0,0,4])))

            s1 = NumpyScalars([1, 2, 3, 4])
            s2 = NumpyScalars([5, 6, 7, 8])
            g = s1.conditional_le(2, s2, s1)
            self.assertTrue(np.all(np.equal(g.v, [5, 6, 3, 4])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsConditional_lt(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s.conditional_lt(3, 0, s)
            self.assertTrue(np.all(np.equal(g.v, [0,0,3,4])))

            s1 = NumpyScalars([1, 2, 3, 4])
            s2 = NumpyScalars([5, 6, 7, 8])
            g = s1.conditional_lt(2, s2, s1)
            self.assertTrue(np.all(np.equal(g.v, [5, 2, 3, 4])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsConditional_gt(self):

        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s.conditional_gt(3, 0, s)
            self.assertTrue(np.all(np.equal(g.v, [1,2,3,0])))

            s1 = NumpyScalars([1, 2, 3, 4])
            s2 = NumpyScalars([5, 6, 7, 8])
            g = s1.conditional_gt(2, s2, s1)
            self.assertTrue(np.all(np.equal(g.v, [1,2,7,8])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsConditional_ge(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s.conditional_ge(3, 0, s)
            self.assertTrue(np.all(np.equal(g.v, [1, 2, 0, 0])))

            s1 = NumpyScalars([1, 2, 3, 4])
            s2 = NumpyScalars([5, 6, 7, 8])
            g = s1.conditional_ge(2, s2, s1)
            self.assertTrue(np.all(np.equal(g.v, [1, 6, 7, 8])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsConditional_eq(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s.conditional_eq(3, 0, s)
            self.assertTrue(np.all(np.equal(g.v, [1, 2, 0, 4])))

            s1 = NumpyScalars([1, 2, 3, 4])
            s2 = NumpyScalars([5, 6, 7, 8])
            g = s1.conditional_eq(2, s2, s1)
            self.assertTrue(np.all(np.equal(g.v, [1, 6, 3, 4])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsConditional_neq(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1,2,3,4])
            g = s.conditional_neq(3, 0, s)
            self.assertTrue(np.all(np.equal(g.v, [0, 0, 3, 0])))

            s1 = NumpyScalars([1, 2, 3, 4])
            s2 = NumpyScalars([5, 6, 7, 8])
            g = s1.conditional_neq(2, s2, s1)
            self.assertTrue(np.all(np.equal(g.v, [5, 2, 7, 8])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsIsBool(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1, 0, 1, 0])
            g = s.isBool
            self.assertTrue(g)

            s = NumpyScalars([2, 0, 1, 0])
            g = s.isBool
            self.assertFalse(g)

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsToIntegers(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1.2, 0.6, 1.5, 0.9])
            g = s.toIntegers()
            self.assertTrue(np.all(np.equal(g.v, [1, 0, 1, 0])))
            self.assertTrue(np.equal(np.dtype("int32"), g.v.dtype))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsLogical_and(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1, 0, 1, 0])
            g = s.logical_and([1, 1, 1, 0])
            self.assertTrue(np.all(np.equal(g.v, [1, 0, 1, 0])))

            s = NumpyScalars([1, 0, 1, 0])
            g = s.logical_and(1)
            self.assertTrue(np.all(np.equal(g.v, [1, 0, 1, 0])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsLogical_or(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1, 0, 1, 0])
            g = s.logical_or([0, 1, 1, 0])
            self.assertTrue(np.all(np.equal(g.v, [1, 1, 1, 0])))

            s = NumpyScalars([1, 0, 1, 0])
            g = s.logical_or(1)
            self.assertTrue(np.all(np.equal(g.v, [1, 1, 1, 1])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsLogical_xor(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([1, 0, 1, 0])
            g = s.logical_xor([0, 1, 1, 0])
            self.assertTrue(np.all(np.equal(g.v, [1, 1, 0, 0])))

            s = NumpyScalars([1, 0, 1, 0])
            g = s.logical_xor(1)
            self.assertTrue(np.all(np.equal(g.v, [0, 1, 0, 1])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testNumpyAdvancedIndexing(self):
        a = np.array([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])
        i = np.array([0,1])
        j = np.array([2,2])
        k = np.array([2,0])
        deltas = np.array([45, 1102])
        a[k, j, i] = deltas
        #print(a)

    def testNumpyZeroCreation(self):
        a = np.zeros((2, 3, 4))
        #print(a)
        #print(a[1, 1, 3])