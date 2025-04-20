import os
import traceback
import unittest

import numpy as np

from pytissueoptics import Cuboid, ScatteringMaterial, ScatteringScene, Vector
from pytissueoptics.rayscattering.opencl import OPENCL_OK
from pytissueoptics.rayscattering.opencl.CLPhotons import CLScene
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.tests.opencl.src.CLObjects import RayCL
from pytissueoptics.rayscattering.tests.opencl.src.testCLFresnel import IntersectionCL


@unittest.skipIf(not OPENCL_OK, 'OpenCL device not available.')
class TestCLIntersection(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "intersection.c")
        self.program = CLProgram(sourcePath)

    def testRayIntersection(self):
        N = 1
        _scene = self._getTestScene()
        clScene = CLScene(_scene, nWorkUnits=1)

        rayLength = 10
        rayOrigin = [0, 0, -7]
        rays = RayCL(origins=np.full((N, 3), rayOrigin),
                     directions=np.full((N, 3), [0, 0, 1]),
                     lengths=np.full(N, rayLength))
        intersections = IntersectionCL(skipDeclaration=True)

        try:
            self.program.launchKernel("findIntersections", N=N, arguments=[rays, clScene.nSolids,
                                                                           clScene.solids, clScene.surfaces,
                                                                           clScene.triangles, clScene.vertices,
                                                                           clScene.solidCandidates, intersections])
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
