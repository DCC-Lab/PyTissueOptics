from typing import List, NamedTuple

import numpy as np

from .CLObject import CLObject, cl

SurfaceCLInfo = NamedTuple(
    "SurfaceInfo",
    [
        ("firstPolygonID", int),
        ("lastPolygonID", int),
        ("insideMaterialID", int),
        ("outsideMaterialID", int),
        ("insideSolidID", int),
        ("outsideSolidID", int),
        ("toSmooth", bool),
    ],
)


class SurfaceCL(CLObject):
    STRUCT_NAME = "Surface"
    STRUCT_DTYPE = np.dtype(
        [
            ("firstPolygonID", cl.cltypes.uint),
            ("lastPolygonID", cl.cltypes.uint),
            ("insideMaterialID", cl.cltypes.uint),
            ("outsideMaterialID", cl.cltypes.uint),
            ("insideSolidID", cl.cltypes.int),
            ("outsideSolidID", cl.cltypes.int),
            ("toSmooth", cl.cltypes.uint),
        ]
    )

    def __init__(self, surfacesInfo: List[SurfaceCLInfo]):
        self._surfacesInfo = surfacesInfo
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._surfacesInfo), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, surfaceInfo in enumerate(self._surfacesInfo):
            buffer[i]["firstPolygonID"] = np.uint32(surfaceInfo.firstPolygonID)
            buffer[i]["lastPolygonID"] = np.uint32(surfaceInfo.lastPolygonID)
            buffer[i]["insideMaterialID"] = np.uint32(surfaceInfo.insideMaterialID)
            buffer[i]["outsideMaterialID"] = np.uint32(surfaceInfo.outsideMaterialID)
            buffer[i]["insideSolidID"] = np.int32(surfaceInfo.insideSolidID)
            buffer[i]["outsideSolidID"] = np.int32(surfaceInfo.outsideSolidID)
            buffer[i]["toSmooth"] = np.uint32(surfaceInfo.toSmooth)
        return buffer
