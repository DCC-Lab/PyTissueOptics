import unittest

from pytissueoptics import Geometry, Photon, Material


class TestGeometryIT(unittest.TestCase):

    def testCreateGeometry(self):
        geometry = Geometry()
        self.assertIsNotNone(geometry)

    def testGivenNoMaterial_whenPropagatePhoton_shouldRaiseError(self):
        geometry = Geometry()
        photon = Photon()

        with self.assertRaises(AttributeError):
            geometry.propagate(photon)

    def testCreateMaterial(self):
        material = Material()
        self.assertIsNotNone(material)

    @unittest.skip("Default material will propagate the photon indefinitely.")
    def testGivenDefaultMaterial_whenPropagatePhoton_shouldPropagateIndefinitely(self):
        material = Material()
        geometry = Geometry(material=material)
        photon = Photon()

        geometry.propagate(photon)

    def testWhenPropagatePhoton_shouldPropagateThePhotonUntilDead(self):
        aMaterial = Material(mu_s=2, mu_a=2, g=0.8, index=1.2)
        geometry = Geometry(material=aMaterial)
        photon = Photon()

        geometry.propagate(photon)

        self.assertFalse(photon.isAlive)

    def testWhenPropagatePhoton(self):
        aMaterial = Material(mu_s=2, mu_a=2, g=0.8, index=1.2)
        geometry = Geometry(material=aMaterial)
        photon = Photon()
        print(photon.origin)
        geometry.propagate(photon)

        print(photon.origin)
# + test geometry with Stats?
