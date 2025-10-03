import unittest
import numpy as np

from pytissueoptics.rayscattering.photon import Photon
# from pytissueoptics.scene.geometry import Vector
from pytissueoptics.rayscattering.compactstructures import CompactRawObject, CompactObject, CompactField, CompactObjects, dtype_vector
from pytissueoptics.scene.geometry import Environment, Vector, CompactVector
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.materials import ScatteringMaterial


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
            def __init__(self, compact_objects, index):
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



    def test_position_direction_environment_assignement(self):
        class CompactPhoton(CompactObject, Photon):
            def __init__(self, compact_objects, index):
                CompactObject.__init__(self, compact_objects, index)
                Photon.__init__(self, Vector(0,0,0), Vector(0,0,1))

                self.energy = 0
                self.polarisation = (1,0,0)

            def set_environment(self, value):
                self.environment = value

        dtype_photon_custom = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32), ("u", np.float32),("v", np.float32),("w", np.float32), ("weight", np.float32), ('environment',np.uint) ])
        photons = CompactObjects(max_count=10, dtype=dtype_photon_custom)
        photons.return_type = CompactPhoton

        for i, photon in enumerate(photons):
            photon.direction =  Vector(0,0,1)
            photon.er =  Vector(1,0,0)
            photon.weight =  1

        scene = ScatteringScene(solids=[], worldMaterial=ScatteringMaterial(mu_s=100, mu_a=0.1, g=0, n=1.0))

        for i, photon in enumerate(photons):
            photon.setContext(scene.getEnvironmentAt(photon.position))
            photon.propagate()
            print(photon.position, photon.direction)


if __name__ == "__main__":
    unittest.main()
