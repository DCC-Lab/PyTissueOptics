from typing import List

import numpy as np

from pytissueoptics.scene.geometry import Vector, Rotation, BoundingBox, utils
from pytissueoptics.scene.solids import Solid


class SolidGroup:
    def __init__(self, solids: List[Solid], position: Vector = Vector(0, 0, 0), material=None, label: str = "solidGroup"):
        self._solids = solids
        self._solidKeys = self.refreshSolidKeys()
        self._position = self.getCentroid()
        self.translateTo(position)
        self._position = position
        self._orientation: Rotation = Rotation()
        self._bbox = None
        self._label = label
        self._material = material
        if self._material is not None:
            self._setMaterial()
        self._resetBoundingBox()

    @property
    def position(self) -> Vector:
        return self._position

    def getCentroid(self) -> Vector:
        vertexSum = Vector(0, 0, 0)
        for solid in self._solids:
            vertexSum += solid.position
        return vertexSum / (len(self._solids))

    def getBoundingBox(self) -> BoundingBox:
        return self._bbox

    def _resetBoundingBox(self):
        self._bbox = self._solids[0].getBoundingBox()
        for solid in self._solids:
            self._bbox.extendTo(solid.getBoundingBox())

    def _setMaterial(self):
        for solid in self._solids:
            solid.setMaterial(self._material)

    def refreshSolidKeys(self) -> List[str]:
        return [solid.getLabel() for solid in self._solids]

    def getSolid(self, label: str) -> Solid:
        return self._solids[self._solidKeys.index(label)]

    def getSolids(self) -> List[Solid]:
        return self._solids

    def translateTo(self, position):
        if position == self._position:
            return
        translationVector = position - self._position
        for solid in self._solids:
            solid.translateBy(translationVector)
        self._position = position
        self._resetBoundingBox()

    def translateBy(self, translationVector: Vector):
        for solid in self._solids:
            solid.translateBy(translationVector)
        self._position += translationVector
        self._resetBoundingBox()

    def rotate(self, xTheta=0, yTheta=0, zTheta=0, rotationCenter: Vector = None):
        rotation = Rotation(xTheta, yTheta, zTheta)
        if rotationCenter is None:
            rotationCenter = self._position
        for solid in self._solids:
            solid.rotate(xTheta, yTheta, zTheta, rotationCenter)

        if rotationCenter is not None or rotationCenter != self.position:
            verticesArrayAtOrigin = np.array([self._position.array]) - rotationCenter.array
            rotatedVerticesArrayAtOrigin = utils.rotateVerticesArray(verticesArrayAtOrigin, rotation)
            rotatedVerticesArray = rotatedVerticesArrayAtOrigin + rotationCenter.array
            self._position = Vector(*rotatedVerticesArray[0])
        self._resetBoundingBox()












