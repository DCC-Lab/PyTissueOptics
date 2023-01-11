from typing import Union, List, Tuple

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.views import ViewGroup, View2D, Direction


class ViewFactory:
    def __init__(self, scene: RayScatteringScene, defaultBinSize: Union[float, Tuple[float, float, float]]):
        self._scene = scene

        self._defaultBinSize3D = defaultBinSize
        if isinstance(self._defaultBinSize3D, float):
            self._defaultBinSize3D = [defaultBinSize] * 3

    def build(self, views: Union[ViewGroup, List[View2D]]) -> List[View2D]:
        if views is None:
            return []
        if isinstance(views, ViewGroup):
            views = self._createFromGroup(views)

        for view in views:
            self._setContext(view)

        return views

    def _createFromGroup(self, viewGroup: ViewGroup):
        # todo: correctly implement all ViewGroups with bitflag logic.
        views = []
        if ViewGroup.SCENE in viewGroup:
            views += self._getDefaultViewsXYZ()

        if ViewGroup.SOLIDS in viewGroup:
            for solid in self._scene.solids:
                views += self._getDefaultViewsXYZ(solidLabel=solid.getLabel())

        if ViewGroup.SURFACES_ENTERING in viewGroup:
            raise NotImplementedError("Default views for ViewGroup.SURFACES not implemented.".format(viewGroup))

        if ViewGroup.SURFACES_LEAVING in viewGroup:
            raise NotImplementedError("Default views for ViewGroup.SURFACES not implemented.".format(viewGroup))

        return views

    @staticmethod
    def _getDefaultViewsXYZ(solidLabel: str = None) -> List[View2D]:
        return [View2D(Direction.X_POS, Direction.Z_POS, solidLabel=solidLabel),
                View2D(Direction.Y_NEG, Direction.Z_POS, solidLabel=solidLabel),
                View2D(Direction.Z_POS, Direction.X_NEG, solidLabel=solidLabel)]

    def _setContext(self, view: View2D):
        if view.solidLabel:
            solid = self._scene.getSolid(view.solidLabel)
            limits3D = solid.getBoundingBox().xyzLimits
        else:
            limits3D = self._scene.getBoundingBox().xyzLimits
        limits3D = [(d[0], d[1]) for d in limits3D]

        limits = (limits3D[view.axisU], limits3D[view.axisV])
        binSize = (self._defaultBinSize3D[view.axisU], self._defaultBinSize3D[view.axisV])
        view.setContext(limits=limits, binSize=binSize)



# todo: this scene dependence is not ideal ...
# init:
    # need bbox of scene
    # bbox of each solid
    # surface labels of each solid
    # todo: 1 or 3 views per surface?
# _buildGroup (or build):
    # todo: set View name with view group name and other related info?
    #  Better, only set view group. Then create getName inside View2D which is dynamic and include current projection direction, view type, viewGroup, interaction keys
