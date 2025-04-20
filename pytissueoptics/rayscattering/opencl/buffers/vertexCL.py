from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vertex

from .CLObject import CLObject, cl


class VertexCL(CLObject):
    STRUCT_NAME = "Vertex"
    STRUCT_DTYPE = np.dtype([("position", cl.cltypes.float3), ("normal", cl.cltypes.float3)])

    def __init__(self, vertices: List[Vertex]):
        self._vertices = vertices
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._vertices), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, vertex in enumerate(self._vertices):
            buffer[i]["position"][0] = np.float32(vertex.x)
            buffer[i]["position"][1] = np.float32(vertex.y)
            buffer[i]["position"][2] = np.float32(vertex.z)
            if vertex.normal is not None:
                buffer[i]["normal"][0] = np.float32(vertex.normal.x)
                buffer[i]["normal"][1] = np.float32(vertex.normal.y)
                buffer[i]["normal"][2] = np.float32(vertex.normal.z)
        return buffer
