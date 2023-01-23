from typing import List

from pytissueoptics.rayscattering.materials.scatteringMaterial import ScatteringMaterial
from pytissueoptics.rayscattering.opencl.buffers.CLObject import *


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
        # todo: there might be a way to abstract both struct and buffer under a single def (DRY, PO)
        #  the cl.types above are actually np types. so we could extract clTypeX and do clTypeX(mat.propertyX) ...
        #  except the float3 thing maybe...
        buffer = np.empty(len(self._materials), dtype=self._dtype)
        for i, material in enumerate(self._materials):
            buffer[i]["mu_s"] = np.float32(material.mu_s)
            buffer[i]["mu_a"] = np.float32(material.mu_a)
            buffer[i]["mu_t"] = np.float32(material.mu_t)
            buffer[i]["g"] = np.float32(material.g)
            buffer[i]["n"] = np.float32(material.n)
            buffer[i]["albedo"] = np.float32(material.getAlbedo())
        return buffer
