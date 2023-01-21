import copy

import numpy as np


class ListArrayContainer:
    def __init__(self):
        self._list = None
        self._array = None

    def __len__(self):
        length = 0
        if self._list is not None:
            length += len(self._list)
        if self._array is not None:
            length += self._array.shape[0]
        return length

    @property
    def _width(self):
        if self._list is not None:
            return len(self._list[0])
        elif self._array is not None:
            return self._array.shape[1]
        else:
            return None

    def append(self, item):
        if isinstance(item, list):
            if self._list is None:
                self._list = [copy.deepcopy(item)]
            else:
                self._list.append(item)
        elif isinstance(item, np.ndarray):
            if self._array is None:
                self._array = copy.deepcopy(item)
            else:
                self._array = np.concatenate((self._array, item), axis=0)

    def extend(self, other: 'ListArrayContainer'):
        if self._list is None:
            self._list = copy.deepcopy(other._list)
        elif other._list is not None:
            self._list.extend(other._list)

        if self._array is None:
            self._array = copy.deepcopy(other._array)
        elif other._array is not None:
            self._array = np.concatenate((self._array, other._array), axis=0)

    def getData(self):
        if self._list is not None and self._array is not None:
            listArray = np.array(self._list)
            if listArray.shape[1] == self._array.shape[1]:
                mergedData = np.concatenate((listArray, self._array), axis=0)
            else:
                raise ValueError("Cannot merge list data and array data with mismatched column count.")
        elif self._list is None:
            mergedData = self._array
        elif self._array is None:
            mergedData = np.array(self._list)
        else:
            mergedData = None
        return mergedData
