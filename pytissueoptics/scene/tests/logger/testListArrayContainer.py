import unittest

import numpy as np

from pytissueoptics.scene.logger.listArrayContainer import ListArrayContainer


class TestListArrayContainer(unittest.TestCase):
    def setUp(self):
        self.listArrayContainer = ListArrayContainer()
        self.otherListArrayContainer = ListArrayContainer()

    def testShouldHaveLengthOfZero(self):
        self.assertEqual(0, len(self.listArrayContainer))

    def testShouldInitializeDataToNone(self):
        self.assertIsNone(self.listArrayContainer.list)
        self.assertIsNone(self.listArrayContainer.array)
        self.assertIsNone(self.listArrayContainer.mergedData)

    def testWhenAppendingList_shouldSetList(self):
        self.listArrayContainer.append([1, 2, 3])
        self.assertEqual([[1, 2, 3]], self.listArrayContainer.list)

    def testWhenAppendingArray_shouldSetArray(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.array))

    def testWhenMergingArrayData_shouldCreateMergedDataAsArrayType(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.listArrayContainer.merge()
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.mergedData))

    def testWhenMergingListData_shouldCreateMergedDataAsArrayType(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.merge()
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.mergedData))

    def testWhenMergingData_shouldCreateMergedDataAsArrayType(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6]]))

        self.listArrayContainer.merge()

        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [4, 5, 6]]), self.listArrayContainer.mergedData))

    def testWhenExtendingWithListData_shouldOnlyExtendThisContainer(self):
        self.listArrayContainer.append([1, 2, 3])
        self.otherListArrayContainer.append([4, 5, 6])

        self.listArrayContainer.extend(self.otherListArrayContainer)

        self.assertEqual([[1, 2, 3], [4, 5, 6]], self.listArrayContainer.list)
        self.assertEqual([[4, 5, 6]], self.otherListArrayContainer.list)
        self.assertIsNone(self.otherListArrayContainer.array)

    def testWhenExtendingWithArrayData_shouldOnlyExtendThisContainer(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.otherListArrayContainer.append(np.array([[4, 5, 6]]))

        self.listArrayContainer.extend(self.otherListArrayContainer)

        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [4, 5, 6]]), self.listArrayContainer.array))
        self.assertTrue(np.array_equal(np.array([[4, 5, 6]]), self.otherListArrayContainer.array))
        self.assertIsNone(self.otherListArrayContainer.list)

    def testWhenExtending_shouldOnlyExtendThisContainer(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6]]))
        self.otherListArrayContainer.append([7, 8, 9])
        self.otherListArrayContainer.append(np.array([[10, 11, 12]]))

        self.listArrayContainer.extend(self.otherListArrayContainer)
        self.listArrayContainer.merge()

        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6], [10, 11, 12]]),
                                       self.listArrayContainer.mergedData))
        self.assertEqual([[7, 8, 9]], self.otherListArrayContainer.list)
        self.assertTrue(np.array_equal(np.array([[10, 11, 12]]), self.otherListArrayContainer.array))

    def testWhenExtendingAnEmptyArray_secondArrayShouldNotHaveReference(self):
        self.otherListArrayContainer.append(np.array([[4, 5, 6]]))
        self.listArrayContainer.extend(self.otherListArrayContainer)
        thirdListArrayContainer = ListArrayContainer()
        thirdListArrayContainer.append(np.array([[7, 8, 9]]))

        self.listArrayContainer.extend(thirdListArrayContainer)

        self.assertTrue(np.array_equal(self.otherListArrayContainer.array, np.array([[4, 5, 6]])))
