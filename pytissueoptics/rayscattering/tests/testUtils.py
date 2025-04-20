import unittest

import numpy as np

from pytissueoptics.rayscattering.utils import labelContained, labelsEqual, logNorm


class TestLogNorm(unittest.TestCase):
    def testGivenConstantArray_shouldWarnAndReturnNANs(self):
        data = np.ones((10, 10))
        with self.assertWarns(Warning):
            result = logNorm(data)
        self.assertTrue(np.isnan(result).all())

    def testGivenArray_shouldReturnLogNormalizedArray(self):
        data = np.exp(np.arange(1, 6))
        result = logNorm(data)
        expected = np.linspace(0, 1, 5)
        self.assertTrue(np.allclose(result, expected, atol=1e-5))


class TestLabelsEqual(unittest.TestCase):
    def testGivenEqualLabels_shouldReturnTrue(self):
        self.assertTrue(labelsEqual("a", "a"))

    def testGivenDifferentLabels_shouldReturnFalse(self):
        self.assertFalse(labelsEqual("a", "b"))

    def testGivenDifferentCaseLabels_shouldReturnTrue(self):
        self.assertTrue(labelsEqual("a", "A"))

    def testGivenOneLabelNone_shouldReturnFalse(self):
        self.assertFalse(labelsEqual("a", None))
        self.assertFalse(labelsEqual(None, "a"))

    def testGivenBothLabelsNone_shouldReturnTrue(self):
        self.assertTrue(labelsEqual(None, None))


class TestLabelContained(unittest.TestCase):
    def testGivenLabelContained_shouldReturnTrue(self):
        self.assertTrue(labelContained("a", ["a", "b", "c"]))

    def testGivenLabelContainedDifferentCase_shouldReturnTrue(self):
        self.assertTrue(labelContained("a", ["A", "b", "c"]))

    def testGivenLabelNotContained_shouldReturnFalse(self):
        self.assertFalse(labelContained("a", ["b", "c"]))

    def testGivenLabelNone_shouldReturnFalse(self):
        self.assertFalse(labelContained(None, ["a", "b", "c"]))
