import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")

class TestScalars(envtest.PyTissueTestCase):
    def testScalars(self):
        s = Scalars(N=100)
        self.assertIsNotNone(s)
        self.assertFalse(s.all())

    def testScalarsNot(self):
        s = Scalars(N=100)
        self.assertIsNotNone(s)
        s = s.negate()
        self.assertTrue(s.all())

    def testScalarsOrAnd(self):
        s1 = Scalars([True, False, True, False])
        s2 = Scalars([True, False, False, True])
        s3 = (s1 and s2)
        self.assertTrue(s3 == Scalars([True, False, False, False]))


if __name__ == '__main__':
    envtest.main()