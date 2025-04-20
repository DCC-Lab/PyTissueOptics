from typing import List

import numpy as np

from pytissueoptics.rayscattering.materials.scatteringMaterial import ScatteringMaterial

from .CLObject import CLObject, cl


class MaterialCL(CLObject):
    STRUCT_NAME = "Material"
    STRUCT_DTYPE = np.dtype(
            [("mu_s", cl.cltypes.float),
             ("mu_a", cl.cltypes.float),
             ("mu_t", cl.cltypes.float),
             ("g", cl.cltypes.float),
             ("n", cl.cltypes.float),
             ("albedo", cl.cltypes.float)])

    def __init__(self, materials: List[ScatteringMaterial]):
        self._materials = materials
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        buffer = np.empty(len(self._materials), dtype=self._dtype)
        for i, material in enumerate(self._materials):
            buffer[i]["mu_s"] = np.float32(material.mu_s)
            buffer[i]["mu_a"] = np.float32(material.mu_a)
            buffer[i]["mu_t"] = np.float32(material.mu_t)
            buffer[i]["g"] = np.float32(material.g)
            buffer[i]["n"] = np.float32(material.n)
            buffer[i]["albedo"] = np.float32(material.getAlbedo())
        return buffer
