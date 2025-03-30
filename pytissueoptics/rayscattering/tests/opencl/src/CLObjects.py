import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics import Vector
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE

if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None

from pytissueoptics.rayscattering.opencl.buffers import CLObject


class IntersectionCL(CLObject):
    STRUCT_NAME = "Intersection"
    STRUCT_DTYPE = np.dtype([("exists", cl.cltypes.uint),
                             ("distance", cl.cltypes.float),
                             ("position", cl.cltypes.float3),
                             ("normal", cl.cltypes.float3),
                             ("surfaceID", cl.cltypes.uint),
                             ("polygonID", cl.cltypes.uint),
                             ("distanceLeft", cl.cltypes.float),
                             ("isSmooth", cl.cltypes.uint),
                             ("rawNormal", cl.cltypes.float3)])

    def __init__(self, distance: float = 10, position=Vector(0, 0, 0), normal=Vector(0, 0, 1),
                 surfaceID=0, polygonID=0, distanceLeft: float = 0, **kwargs):
        self._distance = distance
        self._position = position
        self._normal = normal
        self._surfaceID = surfaceID
        self._polygonID = polygonID
        self._distanceLeft = distanceLeft

        super().__init__(**kwargs)

    def _getInitialHostBuffer(self) -> np.ndarray:
        buffer = np.empty(1, dtype=self._dtype)
        buffer[0]["exists"] = np.uint32(True)
        buffer[0]["distance"] = np.float32(self._distance)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[0, 2:5] = self._position.array
        buffer[0, 6:9] = self._normal.array
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        buffer[0]["surfaceID"] = np.uint32(self._surfaceID)
        buffer[0]["polygonID"] = np.uint32(self._polygonID)
        buffer[0]["distanceLeft"] = np.float32(self._distanceLeft)
        return buffer


class RayCL(CLObject):
    STRUCT_NAME = "Ray"
    STRUCT_DTYPE = np.dtype([("origin", cl.cltypes.float4),
                             ("direction", cl.cltypes.float4),
                             ("length", cl.cltypes.float)])

    def __init__(self, origins: np.ndarray, directions: np.ndarray, lengths: np.ndarray):
        self._origins = origins
        self._directions = directions
        self._lengths = lengths
        self._N = origins.shape[0]

        super().__init__(skipDeclaration=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._origins
        buffer[:, 4:7] = self._directions
        buffer[:, 8] = self._lengths
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        return buffer
