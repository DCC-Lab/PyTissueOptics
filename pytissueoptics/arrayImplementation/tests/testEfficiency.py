import numpy as np
from numpy.random import rand
import time
import unittest


class TestEfficiency(unittest.TestCase):
    header = 0

    def setUp(self):
        self.N = 1000000
        self.v1 = list(rand(self.N))
        self.v2 = list(rand(self.N))
        self.v3 = list(rand(self.N))
        self.v1n = np.array(self.v1)
        self.v2n = np.array(self.v2)
        self.v3n = np.array(self.v3)
        self.b = 9

        if not self.header:
            self.seriesReportHeader()
            self.header = 1

    def seriesReportHeader(self):
        w = 20
        print(f"\n\n {'='*106} ")
        print("||{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}||".format("Function", "Verification", "ListComp No Zip", "Numpy", "Explicit Zip Loop", width=w))
        print(f" {'-'*106} ")

    @staticmethod
    def report(m: str, arrays, namesAndTimes):
        w = 20
        verif = np.all(np.equal(arrays[0], arrays[1])) and np.all(np.equal(arrays[0], arrays[2]))
        t_1 = namesAndTimes[0][1]
        t_2 = namesAndTimes[1][1]
        t_3 = namesAndTimes[2][1]
        print("||{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}|{:^{width}}||".format(m, str(verif), t_1, t_2, t_3, width=w))

    def testConditionsSeries(self):
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

            self.report("==", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

        with self.subTest("!="):
            t0 = time.time_ns()
            vf1 = [self.v3[i] if self.v1[i] != self.v2[i] else self.b for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.where(self.v1n != self.v2, self.v3, self.b)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, value3 in list(zip(self.v1, self.v2, self.v3)):
                if value1 != value2:
                    vf3.append(value3)
                else:
                    vf3.append(self.b)
            t3 = time.time_ns()

            self.report("!=", [vf1, vf2, vf3],[("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

        with self.subTest("<="):
            t0 = time.time_ns()
            vf1 = [self.v3[i] if self.v1[i] <= self.v2[i] else self.b for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.where(self.v1n <= self.v2, self.v3, self.b)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, value3 in list(zip(self.v1, self.v2, self.v3)):
                if value1 <= value2:
                    vf3.append(value3)
                else:
                    vf3.append(self.b)
            t3 = time.time_ns()

            self.report("<=", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

        with self.subTest(">="):
            t0 = time.time_ns()
            vf1 = [self.v3[i] if self.v1[i] >= self.v2[i] else self.b for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.where(self.v1n >= self.v2, self.v3, self.b)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, value3 in list(zip(self.v1, self.v2, self.v3)):
                if value1 >= value2:
                    vf3.append(value3)
                else:
                    vf3.append(self.b)
            t3 = time.time_ns()

            self.report(">=", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

        with self.subTest("<"):
            t0 = time.time_ns()
            vf1 = [self.v3[i] if self.v1[i] < self.v2[i] else self.b for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.where(self.v1n < self.v2, self.v3, self.b)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, value3 in list(zip(self.v1, self.v2, self.v3)):
                if value1 < value2:
                    vf3.append(value3)
                else:
                    vf3.append(self.b)
            t3 = time.time_ns()

            self.report("<", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

        with self.subTest(">"):
            t0 = time.time_ns()
            vf1 = [self.v3[i] if self.v1[i] > self.v2[i] else self.b for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.where(self.v1n > self.v2, self.v3, self.b)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, value3 in list(zip(self.v1, self.v2, self.v3)):
                if value1 > value2:
                    vf3.append(value3)
                else:
                    vf3.append(self.b)
            t3 = time.time_ns()

            self.report(">", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.where", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

    def testArithmeticSeries(self):
        with self.subTest("+"):
            t0 = time.time_ns()
            vf1 = [self.v1[i]+self.v2[i] for i in range(len(self.v1))]
            t1 = time.time_ns()
            vf2 = np.add(self.v1n, self.v2n)
            t2 = time.time_ns()
            vf3 = []
            for value1, value2, in list(zip(self.v1, self.v2)):
                vf3.append(value1+value2)
            t3 = time.time_ns()

            self.report("+", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                        ("numpy.add", str((t2 - t1) / 1000000) + "ms"),
                                                        ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

            self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

            with self.subTest("-"):
                t0 = time.time_ns()
                vf1 = [self.v1[i] - self.v2[i] for i in range(len(self.v1))]
                t1 = time.time_ns()
                vf2 = np.subtract(self.v1n, self.v2n)
                t2 = time.time_ns()
                vf3 = []
                for value1, value2, in list(zip(self.v1, self.v2)):
                    vf3.append(value1 - value2)
                t3 = time.time_ns()

                self.report("-", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                           ("numpy.subtract", str((t2 - t1) / 1000000) + "ms"),
                                                           ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

                self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

            with self.subTest("*"):
                t0 = time.time_ns()
                vf1 = [self.v1[i] * self.v2[i] for i in range(len(self.v1))]
                t1 = time.time_ns()
                vf2 = np.multiply(self.v1n, self.v2n)
                t2 = time.time_ns()
                vf3 = []
                for value1, value2, in list(zip(self.v1, self.v2)):
                    vf3.append(value1 * value2)
                t3 = time.time_ns()

                self.report("*", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                           ("numpy.multiply", str((t2 - t1) / 1000000) + "ms"),
                                                           ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

                self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))

            with self.subTest("/"):
                t0 = time.time_ns()
                vf1 = [self.v1[i] / self.v2[i] for i in range(len(self.v1))]
                t1 = time.time_ns()
                vf2 = np.divide(self.v1n, self.v2n)
                t2 = time.time_ns()
                vf3 = []
                for value1, value2, in list(zip(self.v1, self.v2)):
                    vf3.append(value1 / value2)
                t3 = time.time_ns()

                self.report("/", [vf1, vf2, vf3], [("list comprehension", str((t1 - t0) / 1000000) + "ms"),
                                                           ("numpy.divide", str((t2 - t1) / 1000000) + "ms"),
                                                           ("explicit loop", str((t3 - t2) / 1000000) + "ms")])

                self.assertTrue(np.all(np.equal(vf1, vf2)) and np.all(np.equal(vf2, vf3)))
