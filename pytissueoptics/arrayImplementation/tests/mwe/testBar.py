import numpy as np
import unittest
from bar import *


class TestBar(unittest.TestCase):
    def testIsNullTrue(self):
        v1 = Bar([[0, 0, 0], [0, 0, 0]])
        r = np.all(v1.isNull)
        self.assertTrue(r)