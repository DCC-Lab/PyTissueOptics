import numpy as np

dtype_vector = np.dtype([("x", np.uint),("y", np.float32),("z", np.float64)])

class CompactRawObject:
    def __init__(self, raw_buffer, index, dtype, offset_in_bytes=0, stride_in_bytes = None):
        if stride_in_bytes is None:
            stride_in_bytes = dtype.itemsize

        self._array = np.frombuffer(raw_buffer, dtype=dtype, count=1, offset=offset_in_bytes + stride_in_bytes * index)

    def __getattribute__(self, name):
        if name in ['_array']:
            return super().__getattribute__(name)
        else:
            if self._array.dtype is not None and self._array.dtype.names is not None:
                if name in self._array.dtype.names:
                    return self._array[0][name]

        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in ['_array']:
            super().__setattr__(name, value)
        else:
            if self._array.dtype is not None and self._array.dtype.names is not None:
                if name in self._array.dtype.names:
                    self._array[0][name] = value
                    return

        super().__setattr__(name, value)

    # Convenience functions
    def __repl__(self):
        return str(self)

    def __str__(self):
        return str(dict(zip(self._array.dtype.names,  self._array[0])))

    @property
    def value(self):
        return self._array[0]

    @value.setter
    def value(self, new_value):
        self._array[0] = new_value

    def __getitem__(self, name):
        return self._array[0][name]



class CompactObject(CompactRawObject):

    def __init__(self, compact_objects, index):
        dtype_objects = compact_objects.dtype
        stride_in_bytes = compact_objects.stride_in_bytes

        super().__init__(raw_buffer=compact_objects._array.data, index=index, dtype=dtype_objects, offset_in_bytes=0, stride_in_bytes=stride_in_bytes)

class CompactField(CompactRawObject):

    def __init__(self, compact_objects, index, field):
        dtype_objects = compact_objects.dtype
        stride_in_bytes = compact_objects.stride_in_bytes

        field_offset = dtype_objects.fields[field][1]
        field_dtype = dtype_objects.fields[field][0]
        super().__init__(raw_buffer=compact_objects._array.data, index=index, dtype=field_dtype, offset_in_bytes=field_offset, stride_in_bytes=stride_in_bytes)

class CompactObjects:
    def __init__(self, max_count, dtype, field=None):
        self.return_type = None
        self.field = field

        self._array = np.zeros(shape=(max_count, ), dtype=dtype)
        self._iteration = 0

    @property
    def max_count(self):
        return self._array.shape[0]

    @property
    def stride_in_bytes(self):
        return self._array.itemsize

    @property
    def dtype(self):
        return self._array.dtype

    def __getitem__(self, index):
        if index >= self.max_count or index < 0:
            raise IndexError()
        if self.return_type is None: # Default type 
            if self.field is not None:
                return CompactField(compact_objects=self, index=index, field=self.field)                
            else:
                return CompactObject(compact_objects=self, index=index)
        else:
            if self.field is not None:
                return self.return_type(compact_objects=self, index=index, field=self.field)
            else:
                return self.return_type(compact_objects=self, index=index)

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < self.max_count:
            element = self[self._iteration]
            self._iteration += 1
            return element

        raise StopIteration
