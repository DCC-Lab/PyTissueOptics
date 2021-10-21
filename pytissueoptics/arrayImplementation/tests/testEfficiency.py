import numpy as np
from numpy.random import rand
import time
import unittest


class TestEfficiency(unittest.TestCase):

    def setUp(self):
        self.N = 10000000
        self.v1 = list(rand(self.N))
        self.v2 = list(rand(self.N))
        self.v3 = list(rand(self.N))
        self.v1n = np.array(self.v1)
        self.v2n = np.array(self.v2)
        self.v3n = np.array(self.v3)
        self.b = 9

    @staticmethod
    def report(m: str, arrays, N, namesAndTimes):
        print(f"Function efficiency report of {m} with {N:.2e} random numbers.\n ==========")
        print(f"Verification between answers is: {np.all(np.equal(arrays[0], arrays[1])) and np.all(np.equal(arrays[0], arrays[2]))}")
        for i in range(len(namesAndTimes)):
            print(f"{namesAndTimes[i][0]}::{namesAndTimes[i][1]}")

    def test_Conditions(self):
        with self.subTest("=="):
            t0 = time.time_ns()
            vf1 = [self.v3[i] if self.v1[i] == self.v2[i] else self.b for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.where(self.v1n == self.v2, self.v3, self.b)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, value3 in list(zip(self.v1, self.v2, self.v3)):
                if value1 == value2:
                    vf3.append(value3)
                else:
                    vf3.append(self.b)
            t3 = time.time_ns()

            self.report("==", [vf1, vf2, vf3], self.N, [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

        with self.subTest("dos"):
            self.assertTrue(1)