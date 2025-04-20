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
        self.assertIsNone(self.listArrayContainer.getData())

    def testWhenAppendingList_shouldHaveArrayData(self):
        self.listArrayContainer.append([1, 2, 3])
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.getData()))

    def testWhenAppendingArray_shouldHaveArrayData(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.getData()))

    def testWhenAppendingArrayWithMoreColumns_shouldRaiseException(self):
        self.listArrayContainer.append(np.array([[1, 2, 3]]))

        with self.assertRaises(AssertionError):
            self.listArrayContainer.append(np.array([[4, 5, 6, 7]]))

    def testGivenListAndArrayData_shouldHaveMergedArrayData(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6]]))

        self.assertTrue(np.array_equal(np.array([[1, 2, 3], [4, 5, 6]]), self.listArrayContainer.getData()))

    def testGivenListAndArrayData_shouldHaveLengthEqualToTheTotalNumberOfRows(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6], [7, 8, 9]]))

        self.assertEqual(3, len(self.listArrayContainer))

    def testWhenExtending_shouldOnlyExtendThisContainer(self):
        self.listArrayContainer.append([1, 2, 3])
        self.listArrayContainer.append(np.array([[4, 5, 6]]))
        self.otherListArrayContainer.append([7, 8, 9])
        self.otherListArrayContainer.append(np.array([[10, 11, 12]]))

        self.listArrayContainer.extend(self.otherListArrayContainer)

        self.assertTrue(
            np.array_equal(np.array([[1, 2, 3], [7, 8, 9], [4, 5, 6], [10, 11, 12]]), self.listArrayContainer.getData())
        )
        self.assertTrue(np.array_equal(np.array([[7, 8, 9], [10, 11, 12]]), self.otherListArrayContainer.getData()))

    def testWhenExtendingAnEmptyContainerWithAnother_shouldNotExtendListDataWithAReferenceOfTheOthersListData(self):
        self.otherListArrayContainer.append([4, 5, 6])

        self.listArrayContainer.extend(self.otherListArrayContainer)

        self.otherListArrayContainer.append([7, 8, 9])
        self.assertTrue(np.array_equal([[4, 5, 6]], self.listArrayContainer.getData()))

    def testWhenExtendingAnEmptyContainerWithAnother_shouldNotExtendArrayDataWithAReferenceOfTheOthersArrayData(self):
        self.otherListArrayContainer.append(np.array([[1, 2, 3]]))

        self.listArrayContainer.extend(self.otherListArrayContainer)

        self.otherListArrayContainer.append(np.array([[4, 5, 6]]))
        self.assertTrue(np.array_equal(np.array([[1, 2, 3]]), self.listArrayContainer.getData()))
