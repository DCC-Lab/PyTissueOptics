import os
import traceback
import unittest

import numpy as np

from pytissueoptics import *
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.buffers import *
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.tests.opencl.src.testCLIntersection import RayCL, IntersectionCL
from pytissueoptics.scene.geometry.triangle import Triangle
from pytissueoptics.scene.geometry.vertex import Vertex
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLNormalSmoothing(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "intersection.c")
        self.program = CLProgram(sourcePath)
        self._addMissingDeclarations()

        self.VERTICES = [Vertex(0, 0, 0), Vertex(1, 0, 0), Vertex(1, 1, 0)]
        self.NORMALS = [Vector(-1, -1, 1), Vector(1, -1, 1), Vector(1, 1, 1)]
        for vertex, normal in zip(self.VERTICES, self.NORMALS):
            normal.normalize()
            vertex.normal = normal
        self.TRIANGLE = Triangle(*self.VERTICES)

    def testShouldInterpolateNormalOfVerticesAtIntersectionPosition(self):
        position = Vector(0.75, 0.25, 0)

        smoothNormal = self._getSmoothNormal(position)

        expectedNormal = self.NORMALS[0] + self.NORMALS[1] * 2 + self.NORMALS[2]
        expectedNormal.normalize()
        self.assertEqual(smoothNormal, expectedNormal)

    def testGivenPositionOnVertexShouldReturnVertexNormal(self):
        position = self.VERTICES[0]

        smoothNormal = self._getSmoothNormal(position)

        self.assertEqual(smoothNormal, self.NORMALS[0])

    def testGivenPositionOnEdgeShouldInterpolateBetweenEdgeVertices(self):
        position = Vector(0.5, 0.5, 0)

        smoothNormal = self._getSmoothNormal(position)

        expectedNormal = self.NORMALS[0] + self.NORMALS[2]
        expectedNormal.normalize()
        self.assertEqual(smoothNormal, expectedNormal)

    def testShouldNotSmoothIfItChangesTheSignOfTheDotProductWithTheRayDirection(self):
        position = Vector(1, 1, 0)
        rayDirection = Vector(1, 0, -0.9)
        rayDirection.normalize()

        smoothNormal = self._getSmoothNormal(position, rayDirection)

        self.assertEqual(smoothNormal, self.TRIANGLE.normal)

    def _getSmoothNormal(self, atPosition: Vector, rayDirection: Vector = Vector(0, 0, 1)) -> Vector:
        verticesCL = VertexCL(self.VERTICES)
        triangleInfo = TriangleCLInfo([0, 1, 2], self.TRIANGLE.normal)
        triangleCL = TriangleCL([triangleInfo])
        N = 1
        intersectionCL = IntersectionCL(N)
        intersectionCL.setResults(np.full(N, 0), np.full((N, 3), atPosition.array), np.full((N, 3), self.TRIANGLE.normal.array))
        rayCL = RayCL(origins=np.full((N, 3), [0, 0, 0]),
                      directions=np.full((N, 3), rayDirection.array),
                      lengths=np.full(N, 10))

        try:
            self.program.launchKernel("setSmoothNormals", N=N, arguments=[intersectionCL, triangleCL, verticesCL, rayCL])
        except Exception as e:
            traceback.print_exc(0)

        self.program.getData(intersectionCL)
        smoothNormal = intersectionCL.hostBuffer[0]["normal"]
        return Vector(smoothNormal["x"], smoothNormal["y"], smoothNormal["z"])

    def _addMissingDeclarations(self):
        self.program._include = ''
        missingObjects = [MaterialCL([ScatteringMaterial()]), SurfaceCL([]), SeedCL(1),
                           DataPointCL(1), SolidCandidateCL(1, 1), SolidCL([])]

        for clObject in missingObjects:
            clObject.make(self.program.device)
            self.program.include(clObject.declaration)

if __name__ == "__main__":
    unittest.main()
