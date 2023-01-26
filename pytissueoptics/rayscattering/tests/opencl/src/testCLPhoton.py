import os
import unittest
from dataclasses import dataclass

import numpy as np

from pytissueoptics import Vector, ScatteringMaterial
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from rayscattering.opencl.buffers import *


from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLPhoton(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "propagation.c")
        self.program = CLProgram(sourcePath)

    def _addMissingDeclarations(self):
        clObjects = [MaterialCL([ScatteringMaterial()]), SurfaceCL([]), SeedCL(1), VertexCL([]), DataPointCL(1),
                     SolidCandidateCL(1, 1), TriangleCL([]), SolidCL([])]
        for clObject in clObjects:
            clObject.make(self.program.device)
            self.program.include(clObject.declaration)

    def testWhenMoveBy_shouldMovePhotonByTheGivenDistance(self):
        self._addMissingDeclarations()
        photonBuffer = PhotonCL(positions=np.array([[0, 0, 0]]), directions=np.array([[0, 0, 1]]),
                                materialID=0, solidID=0)
        distance = 10

        self.program.launchKernel(kernelName="moveByKernel", N=1, arguments=[np.float32(distance), photonBuffer, np.int32(0)])

        photonResult = self._getPhotonResult(photonBuffer)
        self.assertEqual(photonResult.position, Vector(0, 0, distance))

    def _getPhotonResult(self, photonBuffer: PhotonCL):
        data = self.program.getData(photonBuffer)[0]
        return PhotonResult(position=Vector(*data[:3]), direction=Vector(*data[4:7]), er=Vector(*data[8:11]),
                            weight=data[12], materialID=data[13], solidID=data[14])


@dataclass
class PhotonResult:
    position: Vector
    direction: Vector
    er: Vector
    weight: float
    materialID: int
    solidID: int
