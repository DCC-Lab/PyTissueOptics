import sys
import unittest
import numpy as np


dtype_vector = np.dtype([("x", np.uint),("y", np.float32),("z", np.float64)])

class CompactRawObject:
    def __init__(self, raw_buffer, index, dtype, offset_in_bytes=0, stride_in_bytes = None):
        if stride_in_bytes is None:
            stride_in_bytes = dtype.itemsize

        self._array = np.frombuffer(raw_buffer, dtype=dtype, count=1, offset=offset_in_bytes + stride_in_bytes * index)

        self.dtype = dtype
        self.raw_buffer = raw_buffer
        self.index = index
        self.offset_in_bytes = offset_in_bytes
        self.as_dict = True
        if stride_in_bytes is None:
            stride_in_bytes = self.dtype.itemsize
        self.stride_in_bytes = stride_in_bytes

        self.as_dict = True

    def __repl__(self):
        return str(self.value)

    def __str__(self):
        if self.as_dict and self.dtype.names is not None:
            return str(dict(zip(self.dtype.names,  self._array[0])))
        return str(self.value)

    @property
    def value(self):
        return self._array[0]

    @value.setter
    def value(self, new_value):
        self._array[0] = new_value

    def __getitem__(self, name):
        return self._array[name]

    def __getattribute__(self, name):
        if name in ['dtype','_array']:
            return super().__getattribute__(name)
        else:
            if self.dtype is not None and self.dtype.names is not None:
                if name in self.dtype.names:
                    return self._array[0][name]
                else:
                    return super().__getattribute__(name)
            else:
                return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name in ['dtype','_array']:
            super().__setattr__(name, value)
        else:
            if self.dtype is not None and self.dtype.names is not None:
                if name in self.dtype.names:
                    self._array[0][name] = value
                else:
                    super().__setattr__(name, value)
            else:
                super().__setattr__(name, value)


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
        self.dtype = dtype
        self.return_type = None
        self.stride_in_bytes = self.dtype.itemsize
        self.max_count = max_count
        self._array = np.zeros(shape=(max_count, ), dtype=self.dtype)
        self.field = field
        self.iteration = 0

    def __getitem__(self, index):
        if index >= self.max_count or index < 0:
            raise IndexError()
        if self.return_type is None: # Default type 
            if self.field is not None:
                return CompactField(compact_objects=self, index=index, field=self.field)                
            else:
                return CompactObject(compact_objects=self, index=index)
        else:
            return self.return_type(compact_objects=self, index=index, field=self.field)

    def __iter__(self):
        self.iteration = 0
        return self

    def __next__(self):
        if self.iteration < self.max_count:
            element = self[self.iteration]
            self.iteration += 1
            return element

        raise StopIteration


class TestCompactObject(unittest.TestCase):
    def test_dtype(self):
        self.assertIsNotNone(dtype_vector)
        dtype = np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})

    def test_memmap(self):
        max_count = 10
        array = np.zeros(shape=(max_count, ), dtype=dtype_vector)
        obj = CompactRawObject(raw_buffer=array.data, index=1, dtype=dtype_vector)
        self.assertIsNotNone(obj)

    def test_astruct(self):
        max_count = 10
        array = np.zeros(shape=(max_count, ), dtype=dtype_vector)
        obj = CompactRawObject(raw_buffer=array.data, index=1, dtype=dtype_vector)
        self.assertIsNotNone(obj._array['x'])

    def test_astruct_property(self):
        max_count = 10
        array = np.zeros(shape=(max_count, ), dtype=dtype_vector)
        obj = CompactRawObject(raw_buffer=array.data, index=1, dtype=dtype_vector)
        self.assertEqual(obj._array['x'], obj.x)
        self.assertEqual(obj._array['x'], obj['x'])
        self.assertEqual(obj['x'], obj.x)

    def test_undefined_property(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector)
        
    def test_objects(self):
        obj = CompactObjects(max_count=100, dtype=dtype_vector)
        self.assertIsNotNone(obj)

    def test_object_in_objects(self):
        obj = CompactObjects(max_count=100, dtype=dtype_vector)
        self.assertIsNotNone(obj)
        self.assertEqual(type(obj[0]), CompactObject)

    def test_struct_in_objects(self):
        obj = CompactObjects(max_count=100, dtype=dtype_vector)
        obj.return_type = None
        self.assertIsNotNone(obj)
        self.assertEqual(type(obj[0]), CompactObject)

    def test_object_in_objects_indexed(self):
        obj = CompactObjects(max_count=100, dtype=dtype_vector)
        self.assertIsNotNone(obj)

        with self.assertRaises(IndexError):
            obj[100]

        with self.assertRaises(IndexError):
            obj[101]

        with self.assertRaises(IndexError):
            obj[-1]

        self.assertIsNotNone(obj[0])
        self.assertIsNotNone(obj[99])

    def test_memory_size(self):
        obj = CompactObjects(max_count=100, dtype=dtype_vector )
        self.assertTrue(obj._array.nbytes > 0)

    def test_fieldobject(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector)
        self.assertTrue(objects._array.nbytes > 0)
        obj = CompactField(objects, index=1, field='x')
        self.assertEqual(obj._array[0], 0)
        self.assertEqual(obj.value, 0)

        obj._array[0] = 1
        self.assertEqual(obj._array[0], 1)
        self.assertEqual(obj.value, 1)

        obj.value = 2
        self.assertEqual(obj._array[0], 2)
        self.assertEqual(obj.value, 2)

    def test_compactobjects_subfield(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector, field='y')
        self.assertTrue(objects._array.nbytes > 0)
        self.assertEqual(objects[0].value, 0)
        objects[1].value = 1
        self.assertEqual(objects[1].value, 1)


    def test_iterate_object(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector)
        for i in range(100):
            objects[i].value = (i,i+1, i+2)

        for i in range(100):
            self.assertEqual(objects[i].value[0], i)
            self.assertEqual(objects[i].value[1], i+1)
            self.assertEqual(objects[i].value[2], i+2)

    def test_iterate_object_with_iterator(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector)
        for i, object in enumerate(objects):
            object.value = (i,i+1, i+2)

        for i in range(100):
            self.assertEqual(objects[i].value[0], i)
            self.assertEqual(objects[i].value[1], i+1)
            self.assertEqual(objects[i].value[2], i+2)

    def test_iterate_field(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector, field='x')
        for i in range(100):
            objects[i].value = i

        for i in range(100):
            self.assertEqual(objects[i].value, i)

    def test_field_value_is_not_array(self):
        objects = CompactObjects(max_count=100,dtype=dtype_vector, field='x')
        
        with self.assertRaises(IndexError):
            objects[0].value[0]

    def test_iterate_object_field_with_iterator(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector, field='z')
        for i, object in enumerate(objects):
            self.assertTrue(isinstance(object.value, np.float64) )

    def test_change_default_struct(self):
        objects = CompactObjects(max_count=100, dtype=np.dtype([("a", np.uint,)]), field='a')
        for i, object in enumerate(objects):
            self.assertTrue(isinstance(object.value, np.uint) )

    def test_change_newcompact_class(self):
        objects = CompactObjects(max_count=100, dtype=dtype_vector)
        self.assertIsNotNone(objects)
        self.assertIsNotNone(objects[0])
        objects[0].y
        objects[0].z
        with self.assertRaises(Exception):
            objects[0].t


    def test_position_direction_weight(self):
        dtype_photon = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32), ("u", np.float32),("v", np.float32),("w", np.float32), ("weight", np.float32) ])
        objects = CompactObjects(max_count=100, dtype=dtype_photon)
        self.assertIsNotNone(objects)

        for i, photon in enumerate(objects):
            photon.y = i

        for i, photon in enumerate(objects):
            self.assertEqual(photon.x, 0)
            self.assertEqual(photon.y, i)

    def test_position_direction_weight_assignement(self):
        dtype_photon = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32), ("u", np.float32),("v", np.float32),("w", np.float32), ("weight", np.float32) ])
        objects = CompactObjects(max_count=100, dtype=dtype_photon)
        self.assertIsNotNone(objects)

        for i, photon in enumerate(objects):
            photon.value = (i,i,2*i, 3*i,4*i,3,4)

        for i, photon in enumerate(objects):
            self.assertEqual(photon.x, i)
            self.assertEqual(photon.y, i)
            self.assertEqual(photon.z, 2*i)
            self.assertEqual(photon.u, 3*i)
            self.assertEqual(photon.v, 4*i)
            self.assertEqual(photon.w, 3)
            self.assertEqual(photon.weight, 4)


    def test_position_direction_environment_assignement(self):
        class CompactPhoton(CompactObject):
            def __init__(self, compact_objects, index, field):
                super().__init__(compact_objects, index)
                self.energy = 0
                self.polarisation = (1,0,0)

            def set_environment(self, value):
                self.environment = value

        dtype_photon_custom = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32), ("u", np.float32),("v", np.float32),("w", np.float32), ("weight", np.float32), ('environment',np.uint) ])
        objects = CompactObjects(max_count=100, dtype=dtype_photon_custom)
        objects.return_type = CompactPhoton
        for photon in objects:
            photon.environment = 123

        for photon in objects:
            self.assertEqual(photon.environment, 123)


if __name__ == "__main__":
    unittest.main()
