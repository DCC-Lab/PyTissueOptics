import unittest

from pytissueoptics import Cuboid, ScatteringMaterial, ScatteringScene, Sphere, Vector
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.CLScene import CLScene
from pytissueoptics.rayscattering.opencl.config.CLConfig import CLConfig


@unittest.skipIf(not OPENCL_AVAILABLE, "OpenCL not available.")
class TestCLScene(unittest.TestCase):
    def testEachTriangleCarriesItsOwnSurfaceID(self):
        # Every triangle must carry the index of the SurfaceCL entry it belongs to, so that the
        # kernel can resolve a polygon -> surface mapping without iterating per-surface ranges
        # (required for BVH leaf traversal).
        material = ScatteringMaterial(0.1, 0.1, 0.8, 1.4)
        cuboid = Cuboid(a=1, b=1, c=1, position=Vector(0, 0, 0), material=material, label="cuboid")
        sphere = Sphere(radius=0.5, order=1, position=Vector(3, 0, 0), material=material, label="sphere")
        scene = ScatteringScene([cuboid, sphere], worldMaterial=ScatteringMaterial())

        clScene = CLScene(scene, nWorkUnits=1)
        device = CLConfig().device
        for buf in (clScene.solids, clScene.surfaces, clScene.triangles):
            buf.make(device)
        triangles = clScene.triangles.hostBuffer
        surfaces = clScene.surfaces.hostBuffer
        solids = clScene.solids.hostBuffer

        for solidIdx in range(int(clScene.nSolids)):
            firstSurfaceID = int(solids[solidIdx]["firstSurfaceID"])
            lastSurfaceID = int(solids[solidIdx]["lastSurfaceID"])
            for surfaceID in range(firstSurfaceID, lastSurfaceID + 1):
                firstPolygonID = int(surfaces[surfaceID]["firstPolygonID"])
                lastPolygonID = int(surfaces[surfaceID]["lastPolygonID"])
                for polygonID in range(firstPolygonID, lastPolygonID + 1):
                    self.assertEqual(
                        surfaceID,
                        int(triangles[polygonID]["surfaceID"]),
                        msg=f"triangle {polygonID} should belong to surface {surfaceID}",
                    )
