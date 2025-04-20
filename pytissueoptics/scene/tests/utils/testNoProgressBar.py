import unittest

from pytissueoptics.scene.utils import noProgressBar


class TestNoProgressBar(unittest.TestCase):
    def testWhenUsingNoProgressBarWithTQDMArguments_shouldWarnAndIterate(self):
        value = 0
        with self.assertWarns(UserWarning):
            pbar = noProgressBar([1, 2, 3], desc="A description")
        for element in pbar:
            value += element
        self.assertEqual(6, value)
