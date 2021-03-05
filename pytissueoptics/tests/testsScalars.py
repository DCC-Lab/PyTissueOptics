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


if __name__ == '__main__':
    envtest.main()