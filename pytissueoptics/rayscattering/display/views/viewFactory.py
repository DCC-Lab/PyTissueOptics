from typing import List, Tuple, Union

import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.display.views.defaultViews import (
    View2DProjectionX,
    View2DProjectionY,
    View2DProjectionZ,
    View2DSurfaceX,
    View2DSurfaceY,
    View2DSurfaceZ,
)
from pytissueoptics.rayscattering.display.views.view2D import View2D, ViewGroup
from pytissueoptics.rayscattering.energyLogging.energyType import EnergyType
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene


class ViewFactory:
    def __init__(
        self,
        scene: ScatteringScene,
        defaultBinSize: Union[float, Tuple[float, float, float]],
        infiniteLimits: tuple,
        energyType=EnergyType.DEPOSITION,
    ):
        self._scene = scene
        self._energyType = energyType

        self._defaultBinSize3D = defaultBinSize
        if isinstance(self._defaultBinSize3D, float):
            self._defaultBinSize3D = (defaultBinSize, defaultBinSize, defaultBinSize)
        self._infiniteLimits = infiniteLimits

    def build(self, views: Union[ViewGroup, List[View2D]]) -> List[View2D]:
        if views is None:
            return []
        if isinstance(views, ViewGroup):
            views = self._createFromGroup(views)

        for view in views:
            self._setContext(view)

        return views

    def _createFromGroup(self, viewGroup: ViewGroup):
        views = []
        if ViewGroup.SCENE in viewGroup:
            views += self._getDefaultViewsXYZ()

        if ViewGroup.SOLIDS in viewGroup:
            for solidLabel in self._scene.getSolidLabels():
                views += self._getDefaultViewsXYZ(solidLabel)

        includeLeaving = ViewGroup.SURFACES_LEAVING in viewGroup
        includeEntering = ViewGroup.SURFACES_ENTERING in viewGroup
        if includeLeaving or includeEntering:
            for solidLabel in self._scene.getSolidLabels():
                for surfaceLabel in self._scene.getSurfaceLabels(solidLabel):
                    views += self._getDefaultSurfaceViews(solidLabel, surfaceLabel, includeLeaving, includeEntering)
                views += self._getDefaultContainedSurfacesViews(solidLabel, includeLeaving, includeEntering)

        return views

    def _getDefaultSurfaceViews(
        self,
        solidLabel: str,
        surfaceLabel: str,
        includeLeaving: bool,
        includeEntering: bool,
        takenFromSolid: str = None,
    ) -> List[View2D]:
        if takenFromSolid is None:
            takenFromSolid = solidLabel
        surfaceNormal = self._getSurfaceNormal(takenFromSolid, surfaceLabel)
        axis = int(np.argmax(np.abs(surfaceNormal)))
        surfaceNormalSign = np.sign(surfaceNormal[axis])

        views = []
        _ViewType = [View2DSurfaceX, View2DSurfaceY, View2DSurfaceZ][axis]
        if includeLeaving:
            views.append(_ViewType(solidLabel=solidLabel, surfaceLabel=surfaceLabel, surfaceEnergyLeaving=True))
        if includeEntering:
            views.append(_ViewType(solidLabel=solidLabel, surfaceLabel=surfaceLabel, surfaceEnergyLeaving=False))
        for view in views:
            if surfaceNormalSign == view.projectionDirection.sign:
                view.flip()
        return views

    def _getSurfaceNormal(self, solidLabel: str, surfaceLabel: str):
        surfacePolygons = self._scene.getSolid(solidLabel).surfaces.getPolygons(surfaceLabel)
        polygonNormals = np.asarray([p.normal.array for p in surfacePolygons])
        surfaceNormal = np.mean(polygonNormals, axis=0)

        if np.linalg.norm(surfaceNormal) < 1e-6:
            # Ill-defined surface normal. Probably some kind of ellipsoid surface. Use Z_NEG normal.
            return [0, 0, -1]

        return surfaceNormal

    def _getDefaultContainedSurfacesViews(
        self, solidLabel: str, includeLeaving: bool, includeEntering: bool
    ) -> List[View2D]:
        views = []
        for containedSolidLabel in self._scene.getContainedSolidLabels(solidLabel):
            for surfaceLabel in self._scene.getSurfaceLabels(containedSolidLabel):
                views += self._getDefaultSurfaceViews(
                    solidLabel, surfaceLabel, includeLeaving, includeEntering, takenFromSolid=containedSolidLabel
                )
        return views

    def _setContext(self, view: View2D):
        if view.solidLabel:
            solid = self._scene.getSolid(view.solidLabel)
            limits3D = solid.getBoundingBox().xyzLimits
            if not self._viewHasValidSurfaceLabel(view):
                utils.warn(
                    "Surface label '{}' not found in solid '{}'. Available surface labels: {}".format(
                        view.surfaceLabel, view.solidLabel, solid.surfaceLabels
                    )
                )
        else:
            sceneBoundingBox = self._scene.getBoundingBox()
            if sceneBoundingBox is None:
                limits3D = self._infiniteLimits
            else:
                limits3D = sceneBoundingBox.xyzLimits
        limits3D = [(d[0], d[1]) for d in limits3D]
        view.setContext(limits3D=limits3D, binSize3D=self._defaultBinSize3D)

    def _viewHasValidSurfaceLabel(self, view) -> bool:
        if view.surfaceLabel is None:
            return True
        solid = self._scene.getSolid(view.solidLabel)
        if view.surfaceLabel in solid.surfaceLabels:
            return True
        for containedSolidLabel in self._scene.getContainedSolidLabels(view.solidLabel):
            if view.surfaceLabel in self._scene.getSurfaceLabels(containedSolidLabel):
                return True

        if view.solidLabel not in view.surfaceLabel:
            view._surfaceLabel = solid.completeSurfaceLabel(view.surfaceLabel)
            return self._viewHasValidSurfaceLabel(view)
        return False

    def _getDefaultViewsXYZ(self, solidLabel: str = None) -> List[View2D]:
        return [
            View2DProjectionX(solidLabel=solidLabel, energyType=self._energyType),
            View2DProjectionY(solidLabel=solidLabel, energyType=self._energyType),
            View2DProjectionZ(solidLabel=solidLabel, energyType=self._energyType),
        ]
