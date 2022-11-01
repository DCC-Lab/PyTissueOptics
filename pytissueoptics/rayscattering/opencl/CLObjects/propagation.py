from typing import List

from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl.CLObjects.CLObject import *
from pytissueoptics.rayscattering.materials.scatteringMaterial import ScatteringMaterial


class PhotonCL(CLObject):
    STRUCT_NAME = "Photon"

    def __init__(self, positions: np.ndarray, directions: np.ndarray,
                 materialID: int, solidID: int):
        self._positions = positions
        self._directions = directions
        self._N = positions.shape[0]
        self._materialID = materialID
        self._solidID = solidID

        photonStruct = np.dtype(
            [("position", cl.cltypes.float4),
             ("direction", cl.cltypes.float4),
             ("er", cl.cltypes.float4),
             ("weight", cl.cltypes.float),
             ("materialID", cl.cltypes.uint),
             ("solidID", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=photonStruct)

    def _getInitialHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._positions
        buffer[:, 4:7] = self._directions
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        buffer["weight"] = 1.0
        buffer["materialID"] = self._materialID
        buffer["solidID"] = self._solidID
        return buffer


class MaterialCL(CLObject):
    STRUCT_NAME = "Material"

    def __init__(self, materials: List[ScatteringMaterial]):
        self._materials = materials

        materialStruct = np.dtype(
            [("mu_s", cl.cltypes.float),
             ("mu_a", cl.cltypes.float),
             ("mu_t", cl.cltypes.float),
             ("g", cl.cltypes.float),
             ("n", cl.cltypes.float),
             ("albedo", cl.cltypes.float)])
        super().__init__(name=self.STRUCT_NAME, struct=materialStruct, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        # todo: there might be a way to abstract both struct and buffer under a single def (DRY, PO)
        buffer = np.empty(len(self._materials), dtype=self._dtype)
        for i, material in enumerate(self._materials):
            buffer[i]["mu_s"] = np.float32(material.mu_s)
            buffer[i]["mu_a"] = np.float32(material.mu_a)
            buffer[i]["mu_t"] = np.float32(material.mu_t)
            buffer[i]["g"] = np.float32(material.g)
            buffer[i]["n"] = np.float32(material.n)
            buffer[i]["albedo"] = np.float32(material.getAlbedo())
        return buffer


class DataPointCL(CLObject):
    STRUCT_NAME = "DataPoint"

    def __init__(self, size: int):
        self._size = size

        dataPointStruct = np.dtype(
            [("delta_weight", cl.cltypes.float),
             ("x", cl.cltypes.float),
             ("y", cl.cltypes.float),
             ("z", cl.cltypes.float),
             ("solidID", cl.cltypes.uint),
             ("surfaceID", cl.cltypes.int)])
        super().__init__(name=self.STRUCT_NAME, struct=dataPointStruct)

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.zeros(self._size, dtype=self._dtype)


class SeedCL(CLObject):
    def __init__(self, size: int):
        self._size = size
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        return np.random.randint(low=0, high=2 ** 32 - 1, size=self._size, dtype=cl.cltypes.uint)
