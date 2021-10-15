import envtest # modifies path
from pytissueoptics import *
import numpy as np
import random
from math import cos

inf = float("+inf")

class TestStats(envtest.PyTissueTestCase):
    def testStatsInit(self):
        self.assertIsNotNone(Stats())

    def testStatRMS(self):
        stats = Stats(min=(-0.5, -0.5, -0.5), max=(0.5, 0.5, 0.5), size=(10, 10, 10), opaqueBoundaries=False)
        self.assertIsNotNone(stats)
        stats.energy[0][0][0] = 1
        stats.energy[1][0][0] = 1
        xWidth, yWidth, zWidth = stats.energyRMSWidths()
        self.assertAlmostEqual(xWidth, 0.05)
        self.assertAlmostEqual(yWidth, 0.0)
        self.assertAlmostEqual(zWidth, 0.0)

if __name__ == '__main__':
    envtest.main()
