import unittest

import numpy as np

from pytissueoptics.scene.logger.listArrayContainer import ListArrayContainer


class TestListArrayContainer(unittest.TestCase):
    def setUp(self):
        self.listArrayContainer = ListArrayContainer()
        self.otherListArrayContainer = ListArrayContainer()

    def test_whenCreatingListArrayContainer_lengthShouldBeZero(self):
        self.assertEqual(0, len(self.listArrayContainer))

    def test_whenNewListArrayContainer_AttributesShouldBeNone(self):
        self.assertIsNone(self.listArrayContainer.list)
        self.assertIsNone(self.listArrayContainer.array)
        self.assertIsNone(self.listArrayContainer.mergedData)

    def test_whenAppendingList_listShouldBeSet(self):
        self.listArrayContainer.append([1, 2, 3])
        self.assertEqual([[1, 2, 3]], self.listArrayContainer.list)

    def test_whenAppendingArray_arrayShouldBeSet(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.array))

    def test_whenMergingData_withListNone_shouldSetToArray(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.listArrayContainer.merge()
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.mergedData))

    def test_whenMergingData_withArrayNone_shouldSetToList(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.merge()
        self.assertEqual([[1, 2, 3]], self.listArrayContainer.mergedData)

    def test_whenMergingData_withBothSet_shouldSetToAnArrayWithBothData(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6]]))
        self.listArrayContainer.merge()
        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [4, 5, 6]]), self.listArrayContainer.mergedData))

    def test_whenExtendingList_listShouldBeExtendedOnFirstObject(self):
        self.listArrayContainer.append([1, 2, 3])
        self.otherListArrayContainer.append([4, 5, 6])
        self.listArrayContainer.extend(self.otherListArrayContainer)
        self.assertEqual([[1, 2, 3], [4, 5, 6]], self.listArrayContainer.list)
        self.assertEqual([[4, 5, 6]], self.otherListArrayContainer.list)
        self.assertIsNone(self.otherListArrayContainer.array)

    def test_whenExtendingArray_arrayShouldBeExtendedOnFirstObject(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.otherListArrayContainer.append(np.array([[4, 5, 6]]))
        self.listArrayContainer.extend(self.otherListArrayContainer)
        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [4, 5, 6]]), self.listArrayContainer.array))
        self.assertTrue(np.array_equal(np.array([[4, 5, 6]]), self.otherListArrayContainer.array))
        self.assertIsNone(self.otherListArrayContainer.list)

    def test_whenExtendingBoth_listShouldBeExtendedOnFirstObject(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6]]))
        self.otherListArrayContainer.append([7, 8, 9])
        self.otherListArrayContainer.append(np.array([[10, 11, 12]]))
        self.listArrayContainer.extend(self.otherListArrayContainer)
        self.listArrayContainer.merge()
        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6], [10, 11, 12]]), self.listArrayContainer.mergedData))
        self.assertEqual([[7, 8, 9]], self.otherListArrayContainer.list)
        self.assertTrue(np.array_equal(np.array([[10, 11, 12]]), self.otherListArrayContainer.array))

