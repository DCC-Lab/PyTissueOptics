import os
import traceback
import unittest

import numpy as np

from pytissueoptics import Cuboid, ScatteringMaterial, ScatteringScene, Sphere, Vector
from pytissueoptics.rayscattering.opencl import OPENCL_OK
from pytissueoptics.rayscattering.opencl.CLPhotons import CLScene
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.tests.opencl.src.CLObjects import RayCL
from pytissueoptics.rayscattering.tests.opencl.src.testCLFresnel import IntersectionCL


@unittest.skipIf(not OPENCL_OK, "OpenCL device not available.")
class TestCLIntersection(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "intersection.c")
        self.program = CLProgram(sourcePath)

    def _runFindIntersections(self, kernelName: str, rays: RayCL, clScene: CLScene, intersections: IntersectionCL):
        baseArgs = [
            rays,
            clScene.nSolids,
            clScene.solids,
            clScene.surfaces,
            clScene.triangles,
            clScene.vertices,
            clScene.solidCandidates,
        ]
        if kernelName == "findIntersectionsBVH":
            baseArgs += [clScene.nNodes, clScene.treeNodes, clScene.leafPolygons]
        baseArgs.append(intersections)
        try:
            self.program.launchKernel(kernelName, N=len(rays._origins), arguments=baseArgs)
        except Exception:
            traceback.print_exc(0)
            raise

    def testRayIntersection(self):
        N = 1
        _scene = self._getTestScene()
        clScene = CLScene(_scene, nWorkUnits=1)

        rayLength = 10
        rayOrigin = [0, 0, -7]
        rays = RayCL(
            origins=np.full((N, 3), rayOrigin), directions=np.full((N, 3), [0, 0, 1]), lengths=np.full(N, rayLength)
        )
        intersections = IntersectionCL(skipDeclaration=True)

        try:
            self.program.launchKernel(
                "findIntersections",
                N=N,
                arguments=[
                    rays,
                    clScene.nSolids,
                    clScene.solids,
                    clScene.surfaces,
                    clScene.triangles,
                    clScene.vertices,
                    clScene.solidCandidates,
                    intersections,
                ],
            )
        except Exception:
            traceback.print_exc(0)

        self.program.getData(clScene.solidCandidates)
        self.program.getData(intersections)

        solidCandidates = clScene.solidCandidates.hostBuffer
        self.assertEqual(solidCandidates[0]["distance"], -1)
        self.assertEqual(solidCandidates[0]["solidID"], 2)
        self.assertEqual(solidCandidates[1]["distance"], 6)
        self.assertEqual(solidCandidates[1]["solidID"], 1)

        rayIntersection = intersections.hostBuffer[0]
        self.assertEqual(rayIntersection["exists"], 1)
        hitPointZ = -1  # taken from scene
        self.assertEqual(rayIntersection["distance"], abs(rayOrigin[2] - hitPointZ))
        self.assertEqual(rayIntersection["position"]["x"], 0)
        self.assertEqual(rayIntersection["position"]["y"], 0)
        self.assertEqual(rayIntersection["position"]["z"], hitPointZ)
        self.assertEqual(rayIntersection["normal"]["x"], 0)
        self.assertEqual(rayIntersection["normal"]["y"], 0)
        self.assertEqual(rayIntersection["normal"]["z"], -1)
        self.assertEqual(rayIntersection["distanceLeft"], rayLength - abs(rayOrigin[2] - hitPointZ))

    def _getTestScene(self):
        material1 = ScatteringMaterial(0.1, 0.8, 0.8, 1.4)
        material2 = ScatteringMaterial(2, 0.8, 0.8, 1.2)
        material3 = ScatteringMaterial(0.5, 0.8, 0.8, 1.3)

        layer1 = Cuboid(a=10, b=10, c=2, position=Vector(0, 0, 0), material=material1, label="Layer 1")
        layer2 = Cuboid(a=10, b=10, c=2, position=Vector(0, 0, 0), material=material2, label="Layer 2")
        tissue = layer1.stack(layer2, "back")
        solid2 = Cuboid(2, 2, 2, position=Vector(10, 0, 0), material=material3)
        scene = ScatteringScene([tissue, solid2], worldMaterial=ScatteringMaterial())
        return scene

    def _getDenseTestScene(self):
        # Several spherical solids of different sizes give the BVH something interesting to do
        # while remaining fully deterministic. Order=2 yields ~320 triangles per sphere.
        material = ScatteringMaterial(0.5, 0.5, 0.8, 1.4)
        outer = Sphere(2.0, order=2, position=Vector(0, 0, 0), material=material, label="outer")
        inner = Sphere(1.0, order=2, position=Vector(0, 0, 0), material=material, label="inner")
        side = Sphere(0.6, order=2, position=Vector(3, 0, 0), material=material, label="side")
        return ScatteringScene([inner, outer, side], worldMaterial=ScatteringMaterial())

    def testBVHFindIntersectionMatchesFlatPathOnDenseScene(self):
        # Parity check: identical rays must produce identical intersections via either path.
        scene = self._getDenseTestScene()
        rng = np.random.default_rng(seed=42)

        # Rays aimed at the scene origin from the +Z hemisphere, plus rays starting inside the
        # outer sphere shooting outward. We need a healthy mix of hits/misses to exercise both
        # the leaf-traversal and the bbox-prune paths.
        outsideOrigins = rng.uniform(low=[-1.5, -1.5, -5.0], high=[1.5, 1.5, -3.0], size=(32, 3)).astype(np.float32)
        outsideDirs = rng.uniform(-0.3, 0.3, size=(32, 3)).astype(np.float32)
        outsideDirs[:, 2] = 1.0
        insideOrigins = rng.uniform(-0.5, 0.5, size=(32, 3)).astype(np.float32)
        insideDirs = rng.uniform(-1.0, 1.0, size=(32, 3)).astype(np.float32)
        origins = np.concatenate([outsideOrigins, insideOrigins], axis=0)
        directions = np.concatenate([outsideDirs, insideDirs], axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        nRays = origins.shape[0]
        lengths = np.full(nRays, 20.0, dtype=np.float32)

        clSceneFlat = CLScene(scene, nWorkUnits=nRays, useBVH=False)
        clSceneBVH = CLScene(scene, nWorkUnits=nRays, useBVH=True)
        self.assertGreater(int(clSceneBVH.nNodes), 1, "BVH should have multiple nodes for dense scene")

        flatRays = RayCL(origins=origins, directions=directions, lengths=lengths)
        bvhRays = RayCL(origins=origins.copy(), directions=directions.copy(), lengths=lengths.copy())
        flatIntersections = self._allocIntersections(nRays)
        bvhIntersections = self._allocIntersections(nRays)

        self._runFindIntersections("findIntersections", flatRays, clSceneFlat, flatIntersections)
        self._runFindIntersections("findIntersectionsBVH", bvhRays, clSceneBVH, bvhIntersections)

        # Pull back the structured arrays so we can address fields by name.
        self.program.getData(flatIntersections, returnData=False)
        self.program.getData(bvhIntersections, returnData=False)
        flat = flatIntersections.hostBuffer
        bvh = bvhIntersections.hostBuffer

        # Sanity: with this geometry we expect at least a quarter of the rays to hit something
        # via the flat path. If not, the test is degenerate.
        flatHitCount = int(np.sum(flat["exists"] == 1))
        self.assertGreater(flatHitCount, nRays // 4, "test rays should produce many hits")

        for i in range(nRays):
            self.assertEqual(int(flat[i]["exists"]), int(bvh[i]["exists"]), msg=f"ray {i}: exists differs")
            if int(flat[i]["exists"]) != 1:
                continue
            self.assertEqual(
                int(flat[i]["polygonID"]), int(bvh[i]["polygonID"]),
                msg=f"ray {i}: polygonID differs (flat={int(flat[i]['polygonID'])} bvh={int(bvh[i]['polygonID'])})",
            )
            self.assertAlmostEqual(
                float(flat[i]["distance"]), float(bvh[i]["distance"]), places=4,
                msg=f"ray {i}: distance differs",
            )

    @staticmethod
    def _allocIntersections(n: int) -> IntersectionCL:
        # IntersectionCL by default constructs a single-element buffer; override to size n.
        intersections = IntersectionCL(skipDeclaration=True)

        def _hostBuf(_self=intersections, _n=n):
            buffer = np.empty(_n, dtype=_self._dtype)
            buffer[:]["exists"] = 0
            buffer[:]["distance"] = 0
            return buffer

        intersections._getInitialHostBuffer = _hostBuf
        return intersections
