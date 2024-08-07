import envtest
import math
import sys
import unittest
import numpy as np
from pytissueoptics.rayscattering.compactphoton import CompactPhoton, CompactPhotons
from pytissueoptics.scene.geometry import Environment, Vector, CompactVector
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.materials import ScatteringMaterial


class TestCompactPhotons(unittest.TestCase):
    def test_dtype(self):
        npVector = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32)])

    def test_compact_vector(self):
        a = np.array([1,2,3,4,5,6], dtype=np.float32)
        v = CompactVector(rawBuffer=a.data, index=0, offset=0, stride=4)
        self.assertEqual(v, Vector(1,2,3))
        v.x=7
        v.y=8
        v.z=9
        self.assertEqual(v, Vector(7,8,9))
        self.assertTrue( all(a==[7,8,9,4,5,6]))

        v2 = CompactVector(rawBuffer=a.data, index=3, offset=0, stride=4)
        self.assertEqual(v2, Vector(4,5,6))
        self.assertTrue( all(a==[7,8,9,4,5,6]))
        v2.x = 10
        self.assertTrue( all(a==[7,8,9,10,5,6]))

        v3 = CompactVector(rawBuffer=a.data, index=0, offset=3*4, stride=4)
        self.assertEqual(v3, Vector(10,5,6))
        v3.y = 11
        self.assertTrue( all(a==[7,8,9,10,11,6]))


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

    def test_propagate_no_profiler(self):
        photons = CompactPhotons(maxCount=10)
        for i, photon in enumerate(photons):
            photon.direction =  Vector(0,0,1)
            photon.er =  Vector(1,0,0)
            photon.weight =  1

        scene = ScatteringScene(solids=[], worldMaterial=ScatteringMaterial(mu_s=100, mu_a=0.1, g=0, n=1.0))

        for i, photon in enumerate(photons):
            photon.setContext(scene.getEnvironmentAt(photon.position))
            photon.propagate()
            print(i)

    # @unittest.skip('No profiling')
    def test_propagate_profiler(self):
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()

        photons = CompactPhotons(maxCount=10)
        for i, photon in enumerate(photons):
            photon.direction =  Vector(0,0,1)
            photon.er =  Vector(1,0,0)
            photon.weight =  1

        scene = ScatteringScene(solids=[], worldMaterial=ScatteringMaterial(mu_s=100, mu_a=0.1, g=0, n=1.0))

        for i, photon in enumerate(photons):
            photon.setContext(scene.getEnvironmentAt(photon.position))
            photon.propagate()

        pr.disable()
        s = io.StringIO()
        sortby = SortKey.TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

if __name__ == "__main__":
    unittest.main()
