import copy
from typing import Optional

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

    def _assertSameWidth(self, data):
        if self._width is None:
            return
        if isinstance(data, list):
            assert len(data) == self._width
        elif isinstance(data, np.ndarray):
            assert data.shape[1] == self._width

    def append(self, item):
        self._assertSameWidth(item)
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

    def extend(self, other: "ListArrayContainer"):
        if self._list is None:
            self._list = copy.deepcopy(other._list)
        elif other._list is not None:
            self._list.extend(other._list)
        if other._array is not None:
            self.append(other._array)

    def getData(self) -> Optional[np.ndarray]:
        if self._list is None and self._array is None:
            return None
        if self._list is None:
            return self._array
        if self._array is None:
            return np.array(self._list)
        mergedData = np.concatenate((np.array(self._list), self._array), axis=0)
        return mergedData
