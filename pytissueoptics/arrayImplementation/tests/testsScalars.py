import unittest
from arrayImplementation.scalars import *


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
            s1 = NumpyScalars([1]*1000)
            s2 = NumpyScalars([2]*1000)
            sf = s1 + s2
            self.assertTrue(np.all(np.equal(sf, 3)))

            s1 = NumpyScalars([1, 2, 3])
            s2 = NumpyScalars([3, 4, 5])
            sf = s1 + s2
            self.assertTrue(np.all(np.equal(sf, [4, 6, 8])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsSubtract(self):
        with self.subTest("NumpyScalar"):
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
                s2 = NumpyScalars([3, 4, 5])
                sf = s1 * s2
                self.assertTrue(np.all(np.equal(sf, [3, 8, 15])))

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass

    def testScalarsTruediv(self):
        with self.subTest("NumpyScalar"):
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

    def testScalarsNot(self):
        with self.subTest("NumpyScalar"):
            s = NumpyScalars([0]*1000)
            self.assertFalse(s.all())
            s = ~s
            self.assertTrue(s.all())

        with self.subTest("CupyScalar"):
            pass

        with self.subTest("OpenclScalar"):
            pass


