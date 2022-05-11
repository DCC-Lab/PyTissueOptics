import numpy as np


class ListArrayContainer:
    def __init__(self):
        self.list = None
        self.array = None
        self.mergedData = None

    def __len__(self):
        length = 0
        if self.list is not None:
            length += len(self.list)
        if self.array is not None:
            length += self.array.shape[0]
        return length

    def append(self, item):
        if isinstance(item, list):
            if self.list is None:
                self.list = [item]
            else:
                self.list.append(item)
        elif isinstance(item, np.ndarray):
            if self.array is None:
                self.array = item
            else:
                self.array = np.concatenate((self.array, item), axis=0)

    def extend(self, other: 'ListArrayContainer'):
        if self.list is None:
            self.list = other.list
        elif other.list is None:
            return
        else:
            self.list.extend(other.list)

        if self.array is None:
            self.array = other.array
        elif other.array is None:
            return
        else:
            self.array = np.concatenate((self.array, other.array), axis=0)

    def merge(self):
        if self.list is not None and self.array is not None:
            listArray = np.array(self.list)
            if listArray.shape[1] == self.array.shape[1]:
                self.mergedData = np.concatenate((listArray, self.array), axis=0)
            else:
                raise ValueError("ListArrayContainer.merge(): Error: listArray.shape != self.array.shape")

        elif self.list is None:
            self.mergedData = self.array

        elif self.array is None:
            self.mergedData = self.list

        else:
            self.mergedData = None
