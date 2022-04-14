from pytissueoptics.scene.solids import Cylinder


class Cone(Cylinder):

    def _getShrinkFactor(self, heightAlong: float) -> float:
        return (self._height - heightAlong) / self._height

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")
