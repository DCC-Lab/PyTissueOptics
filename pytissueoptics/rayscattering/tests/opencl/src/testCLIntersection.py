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

    def _runFindIntersections(
        self,
        kernelName: str,
        rays: RayCL,
        clScene: CLScene,
        intersections: IntersectionCL,
        photonSolidID: int = -1,
        ignoreSolidID: int = 0,
    ):
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
        # photonSolidID/ignoreSolidID flow through as uint to the kernel; the WORLD_SOLID_ID
        # sentinel is -1 so it round-trips as 0xFFFFFFFF and matches the kernel's WORLD checks.
        baseArgs += [np.uint32(photonSolidID & 0xFFFFFFFF), np.uint32(ignoreSolidID), intersections]
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
                    np.uint32(0xFFFFFFFF),  # photonSolidID = -1 (world)
                    np.uint32(0),  # ignoreSolidID = 0 (no ignore)
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

    def _denseTestRays(self, seed: int = 42, nOutside: int = 32, nInside: int = 32):
        rng = np.random.default_rng(seed=seed)
        outsideOrigins = rng.uniform(
            low=[-1.5, -1.5, -5.0], high=[1.5, 1.5, -3.0], size=(nOutside, 3)
        ).astype(np.float32)
        outsideDirs = rng.uniform(-0.3, 0.3, size=(nOutside, 3)).astype(np.float32)
        outsideDirs[:, 2] = 1.0
        insideOrigins = rng.uniform(-0.5, 0.5, size=(nInside, 3)).astype(np.float32)
        insideDirs = rng.uniform(-1.0, 1.0, size=(nInside, 3)).astype(np.float32)
        origins = np.concatenate([outsideOrigins, insideOrigins], axis=0)
        directions = np.concatenate([outsideDirs, insideDirs], axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        lengths = np.full(origins.shape[0], 20.0, dtype=np.float32)
        return origins, directions, lengths

    def _runParity(self, scene, origins, directions, lengths, photonSolidID: int, ignoreSolidID: int, minHits: int):
        """Run findIntersections (flat) and findIntersectionsBVH on the same rays/scene and
        assert the two paths agree on every output field of the resulting Intersection."""
        nRays = origins.shape[0]
        clSceneFlat = CLScene(scene, nWorkUnits=nRays, useBVH=False)
        clSceneBVH = CLScene(scene, nWorkUnits=nRays, useBVH=True)
        self.assertGreater(int(clSceneBVH.nNodes), 1, "BVH should have multiple nodes")

        flatRays = RayCL(origins=origins, directions=directions, lengths=lengths)
        bvhRays = RayCL(origins=origins.copy(), directions=directions.copy(), lengths=lengths.copy())
        flat = self._allocIntersections(nRays)
        bvh = self._allocIntersections(nRays)
        self._runFindIntersections(
            "findIntersections", flatRays, clSceneFlat, flat, photonSolidID, ignoreSolidID
        )
        self._runFindIntersections(
            "findIntersectionsBVH", bvhRays, clSceneBVH, bvh, photonSolidID, ignoreSolidID
        )

        self.program.getData(flat, returnData=False)
        self.program.getData(bvh, returnData=False)
        flatBuf, bvhBuf = flat.hostBuffer, bvh.hostBuffer

        hitCount = int(np.sum(flatBuf["exists"] == 1))
        self.assertGreaterEqual(
            hitCount, minHits, f"degenerate parity test: only {hitCount} hits for nRays={nRays}"
        )

        for i in range(nRays):
            self.assertEqual(int(flatBuf[i]["exists"]), int(bvhBuf[i]["exists"]), msg=f"ray {i}: exists differs")
            if int(flatBuf[i]["exists"]) != 1:
                continue
            # surfaceID equality is the firm invariant: when a ray grazes a triangle edge,
            # the two paths can each pick a different polygon belonging to the same surface
            # (both have equal distance and the strict `<` in _testPolygonIntersection means
            # whichever was tested first wins). Position and distance are still expected to
            # agree to float precision.
            self.assertEqual(
                int(flatBuf[i]["surfaceID"]), int(bvhBuf[i]["surfaceID"]),
                msg=f"ray {i}: surfaceID differs",
            )
            self.assertAlmostEqual(
                float(flatBuf[i]["distance"]), float(bvhBuf[i]["distance"]), places=4,
                msg=f"ray {i}: distance differs",
            )
            for axis in ("x", "y", "z"):
                self.assertAlmostEqual(
                    float(flatBuf[i]["position"][axis]), float(bvhBuf[i]["position"][axis]),
                    places=4, msg=f"ray {i}: position.{axis} differs",
                )
                self.assertAlmostEqual(
                    float(flatBuf[i]["normal"][axis]), float(bvhBuf[i]["normal"][axis]),
                    places=4, msg=f"ray {i}: normal.{axis} differs",
                )
            self.assertAlmostEqual(
                float(flatBuf[i]["distanceLeft"]), float(bvhBuf[i]["distanceLeft"]),
                places=4, msg=f"ray {i}: distanceLeft differs",
            )

    def testBVHFindIntersectionMatchesFlatPathOnDenseScene(self):
        # World photon (photonSolidID=-1, ignoreSolidID=0): the most permissive case, every
        # surface the ray crosses is a candidate.
        scene = self._getDenseTestScene()
        origins, directions, lengths = self._denseTestRays()
        self._runParity(
            scene, origins, directions, lengths,
            photonSolidID=-1, ignoreSolidID=0, minHits=origins.shape[0] // 4,
        )

    def testBVHFindIntersectionMatchesFlatPathForPhotonInsideASolid(self):
        # Photon already inside the inner sphere: the BVH path's same-solid-skip and
        # surface-environment-filter branches must match the flat path. This is the >99%
        # production case (photons spend their lives inside a solid) and was previously
        # unexercised because both test kernels hardcoded photonSolidID=-1.
        scene = self._getDenseTestScene()
        clScene = CLScene(scene, nWorkUnits=1)
        innerSolidID = clScene.getSolidID(scene.solids[0])  # "inner" added first

        # Rays starting near the inner sphere center, shooting in random directions. With a
        # 1.0-radius inner sphere most of these will hit the inner-outer interface.
        rng = np.random.default_rng(seed=7)
        nRays = 64
        origins = rng.uniform(-0.4, 0.4, size=(nRays, 3)).astype(np.float32)
        directions = rng.uniform(-1.0, 1.0, size=(nRays, 3)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        lengths = np.full(nRays, 5.0, dtype=np.float32)
        self._runParity(
            scene, origins, directions, lengths,
            photonSolidID=innerSolidID, ignoreSolidID=0, minHits=nRays // 2,
        )

    def _disjointSpheresScene(self):
        """Two non-overlapping spheres so a world photon can legally hit either, and the
        ignoreSolidID filter is meaningful (it makes one sphere invisible)."""
        material = ScatteringMaterial(0.5, 0.5, 0.8, 1.4)
        sphereA = Sphere(0.8, order=2, position=Vector(-1.5, 0, 0), material=material, label="A")
        sphereB = Sphere(0.8, order=2, position=Vector(1.5, 0, 0), material=material, label="B")
        return ScatteringScene([sphereA, sphereB], worldMaterial=ScatteringMaterial())

    def testBVHFindIntersectionMatchesFlatPathWithIgnoreSolid(self):
        # Non-zero ignoreSolidID exercises the per-polygon skip in _testPolygonIntersection
        # that the flat path implements per-solid via _findBBoxIntersectingSolids. Without
        # this case, the entire ignore branch in the BVH path is dead code from the test
        # suite's perspective.
        scene = self._disjointSpheresScene()
        clScene = CLScene(scene, nWorkUnits=1)
        sphereAID = clScene.getSolidID(scene.solids[0])

        # Rays from -y heading +y, sweeping across x so they pass through either sphere A
        # (centered at x=-1.5) or sphere B (centered at x=+1.5).
        rng = np.random.default_rng(seed=11)
        nRays = 64
        origins = np.zeros((nRays, 3), dtype=np.float32)
        origins[:, 0] = rng.uniform(-2.0, 2.0, size=nRays).astype(np.float32)
        origins[:, 1] = -3.0
        directions = np.zeros((nRays, 3), dtype=np.float32)
        directions[:, 1] = 1.0
        lengths = np.full(nRays, 10.0, dtype=np.float32)

        # Without the ignore filter both spheres are visible; expect plenty of hits.
        self._runParity(
            scene, origins, directions, lengths,
            photonSolidID=-1, ignoreSolidID=0, minHits=nRays // 2,
        )
        # With sphere A ignored, only rays aimed at sphere B should hit. The two paths must
        # agree; sphere B occupies a small fraction of the x-sweep so a handful of hits is
        # enough to demonstrate parity through the ignore branch.
        self._runParity(
            scene, origins, directions, lengths,
            photonSolidID=-1, ignoreSolidID=sphereAID, minHits=8,
        )

    def testBVHFindIntersectionMatchesFlatPathOnStackedSolids(self):
        # Stacked Cuboids share interface polygons across surfaces. Each interface polygon
        # gets its own SurfaceCL entry via _processSurface; the BVH groups polygons by
        # spatial position rather than by the per-surface ranges that the flat path scans.
        # This is the case where an off-by-one in triangles[p].surfaceID would bite.
        scene = self._getTestScene()
        clScene = CLScene(scene, nWorkUnits=1, useBVH=True)
        if int(clScene.nNodes) <= 1:
            self.skipTest("Stacked test scene below BVH threshold; cannot parity-test.")

        rng = np.random.default_rng(seed=3)
        nRays = 64
        origins = rng.uniform(low=[-4, -4, -4], high=[4, 4, -3], size=(nRays, 3)).astype(np.float32)
        directions = np.zeros_like(origins)
        directions[:, 2] = 1.0
        directions[:, 0:2] = rng.uniform(-0.2, 0.2, size=(nRays, 2)).astype(np.float32)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        lengths = np.full(nRays, 30.0, dtype=np.float32)

        self._runParity(
            scene, origins, directions, lengths,
            photonSolidID=-1, ignoreSolidID=0, minHits=nRays // 2,
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
