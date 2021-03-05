import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random

inf = float("+inf")

class TestNumpyScalars(envtest.PyTissueTestCase):
    def testMultiplicationWithNumpyScalars(self):
        pass


if __name__ == '__main__':
    envtest.main()