import envtest
import math
import sys
import unittest
import numpy as np
from pytissueoptics.rayscattering.compactphoton import CompactPhoton, CompactPhotons, CompactVector, Vector
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.materials import ScatteringMaterial


class TestCompactPhotons(unittest.TestCase):
    def test_dtype(self):
        npVector = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32)])

    def test_compact_vector(self):
        a = np.array([1,2,3,4,5,6], dtype=np.float32)
        v = CompactVector(rawBuffer=a.data, index=0, offset=4, stride=4*3)
        print(v)

        v.x=4
        v.y=5
        v.z=6
        print(v)


    def test_init_compact_photons(self):
        self.assertIsNotNone(CompactPhotons(maxCount=1000))

    def test_get_compact_photon_from_photons(self):
        photons = CompactPhotons(maxCount=1000)

        for i in range(photons.maxCount):
            self.assertEqual(photons[i].position, Vector(0,0,0))
            self.assertEqual(photons[i].direction,Vector(0,0,0))
            self.assertEqual(photons[i].er, Vector(0,0,0))
            self.assertEqual(photons[i].weight, 0)

    def test_assign_compact_photon_from_photons(self):
        photons = CompactPhotons(maxCount=1000)

        for i in range(photons.maxCount):
            photons[i].position =  Vector(1,2,3)
            photons[i].direction =  Vector(2*i,2*i,2*i)
            photons[i].er =  Vector(3*i,3*i,3*i)
            photons[i].weight =  i

        for i in range(photons.maxCount):
            self.assertEqual(photons[i].position, Vector(1,2,3))
            self.assertEqual(photons[i].direction, Vector(2*i, 2*i, 2*i))
            self.assertEqual(photons[i].er, Vector(3*i, 3*i, 3*i))
            self.assertEqual(photons[i].weight, i)

    def test_assign_compact_photon_from_photons(self):
        photons = CompactPhotons(maxCount=1000)

        for i, photon in enumerate(photons):
            photon.position =  Vector(i,i,i)
            photon.direction =  Vector(2*i,2*i,2*i)
            photon.er =  Vector(3*i,3*i,3*i)
            photon.weight =  i

        for i in range(photons.maxCount):
            self.assertEqual(photons[i].position, Vector(i,i,i))
            self.assertEqual(photons[i].direction, Vector(2*i, 2*i, 2*i))
            self.assertEqual(photons[i].er, Vector(3*i, 3*i, 3*i))
            self.assertEqual(photons[i].weight, i)

    def test_propagate(self):
        photons = CompactPhotons(maxCount=1000)
        for i, photon in enumerate(photons):
            photon.direction =  Vector(0,0,1)
            photon.er =  Vector(1,0,0)
            photon.weight =  1

        scene = ScatteringScene(solids=[], worldMaterial=ScatteringMaterial(mu_s=100, mu_a=0.1, g=0, n=1.0))

        for i, photon in enumerate(photons):
            photon.setContext(scene.getEnvironmentAt(photon.position))
            photon.propagate()


if __name__ == "__main__":
    unittest.main()
