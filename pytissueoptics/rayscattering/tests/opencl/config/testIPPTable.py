import json
import os
import tempfile
import unittest

from pytissueoptics.rayscattering.opencl.config.IPPTable import DEFAULT_IPP, IPPTable


def tempTablePath(func):
    def wrapper(*args, **kwargs):
        with tempfile.TemporaryDirectory() as tempDir:
            IPPTable.TABLE_PATH = os.path.join(tempDir, "ipp.json")
            func(*args, **kwargs)
    return wrapper


class TestIPPTable(unittest.TestCase):
    @tempTablePath
    def testGivenNoIPPTableFile_shouldCreateAndSetDefaultIPPValues(self):
        self.table = IPPTable()

        self.assertTrue(os.path.exists(IPPTable.TABLE_PATH))
        for key, (N, IPP) in DEFAULT_IPP.items():
            self.assertEqual(IPP, self.table.getIPP(int(key)))

    @tempTablePath
    def testGivenIPPTableFile_shouldLoadIPPValuesFromFile(self):
        with open(IPPTable.TABLE_PATH, "w") as f:
            f.write('{"1234": [40000, 144.571]}')

        self.table = IPPTable()

        self.assertEqual(144.571, self.table.getIPP(1234))

    @tempTablePath
    def testWhenUpdateIPPWithNewSample_shouldUpdateIPPValueWeightedWithPhotonCount(self):
        expHash = 1234
        oldN = 1000
        oldIPP = 100
        with open(IPPTable.TABLE_PATH, "w") as f:
            f.write('{"%d": [%d, %f]}' % (expHash, oldN, oldIPP))
        self.table = IPPTable()

        sampleN = 4000
        sampleIPP = 200
        self.table.updateIPP(expHash, sampleN, sampleIPP)

        expectedIPP = 180
        self.assertEqual(expectedIPP, self.table.getIPP(expHash))

    @tempTablePath
    def testWhenUpdateIPP_shouldSaveIPPTableToFile(self):
        self.table = IPPTable()
        newHash = 1234
        N = 4000
        IPP = 200

        self.table.updateIPP(newHash, N, IPP)

        with open(IPPTable.TABLE_PATH, "r") as f:
            table = json.load(f)
            self.assertEqual(N, table[str(newHash)][0])
            self.assertEqual(IPP, table[str(newHash)][1])

    @tempTablePath
    def testWhenUpdateIPPWithNewHash_shouldContainNewHash(self):
        self.table = IPPTable()
        newHash = 1234
        self.assertFalse(newHash in self.table)

        self.table.updateIPP(newHash, 4000, 200)

        self.assertTrue(newHash in self.table)

    @tempTablePath
    def testWhenGetIPPWithNonExistingHash_shouldReturnNone(self):
        self.table = IPPTable()
        self.assertIsNone(self.table.getIPP(1234))
