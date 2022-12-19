from typing import List, NamedTuple

from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.rayscattering.opencl.CLObjects.CLObject import *


SolidCLInfo = NamedTuple("SolidInfo", [("bbox", BoundingBox),
                                       ("firstSurfaceID", int), ("lastSurfaceID", int)])


class SolidCL(CLObject):
    STRUCT_NAME = "Solid"
    STRUCT_DTYPE = np.dtype(
            [("bbox_min", cl.cltypes.float3),
             ("bbox_max", cl.cltypes.float3),
             ("firstSurfaceID", cl.cltypes.uint),
             ("lastSurfaceID", cl.cltypes.uint)])

    def __init__(self, solidsInfo: List[SolidCLInfo]):
        self._solidsInfo = solidsInfo
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._solidsInfo), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, solidInfo in enumerate(self._solidsInfo):
            buffer[i]["bbox_min"][0] = np.float32(solidInfo.bbox.xMin)
            buffer[i]["bbox_min"][1] = np.float32(solidInfo.bbox.yMin)
            buffer[i]["bbox_min"][2] = np.float32(solidInfo.bbox.zMin)
            buffer[i]["bbox_max"][0] = np.float32(solidInfo.bbox.xMax)
            buffer[i]["bbox_max"][1] = np.float32(solidInfo.bbox.yMax)
            buffer[i]["bbox_max"][2] = np.float32(solidInfo.bbox.zMax)
            buffer[i]["firstSurfaceID"] = np.uint32(solidInfo.firstSurfaceID)
            buffer[i]["lastSurfaceID"] = np.uint32(solidInfo.lastSurfaceID)
        return buffer
