from typing import List, NamedTuple

import numpy as np

from pytissueoptics.scene.geometry import Vector

from .CLObject import CLObject, cl

TriangleCLInfo = NamedTuple("TriangleInfo", [("vertexIDs", list), ("normal", Vector)])


class TriangleCL(CLObject):
    STRUCT_NAME = "Triangle"
    STRUCT_DTYPE = np.dtype(
        [("vertexIDs", cl.cltypes.uint, 3),
         ("normal", cl.cltypes.float3)]
    )

    def __init__(self, trianglesInfo: List[TriangleCLInfo]):
        self._trianglesInfo = trianglesInfo
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._trianglesInfo), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, triangleInfo in enumerate(self._trianglesInfo):
            buffer[i]["vertexIDs"][0] = np.uint32(triangleInfo.vertexIDs[0])
            buffer[i]["vertexIDs"][1] = np.uint32(triangleInfo.vertexIDs[1])
            buffer[i]["vertexIDs"][2] = np.uint32(triangleInfo.vertexIDs[2])
            buffer[i]["normal"][0] = np.float32(triangleInfo.normal.x)
            buffer[i]["normal"][1] = np.float32(triangleInfo.normal.y)
            buffer[i]["normal"][2] = np.float32(triangleInfo.normal.z)
        return buffer
