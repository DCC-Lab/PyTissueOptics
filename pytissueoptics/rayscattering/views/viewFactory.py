from typing import Union, List

from pytissueoptics import RayScatteringScene
from pytissueoptics.rayscattering.views import ViewGroup, View2D, Direction
from pytissueoptics.scene.solids import Solid


class ViewFactory:
    def __init__(self, scene: RayScatteringScene):
        self._scene = scene

    def build(self, views: Union[ViewGroup, List[View2D]]) -> List[View2D]:
        if isinstance(views, ViewGroup):
            return self._buildGroup(views)
        else:
            return self._buildViews(views)

    def _buildGroup(self, viewGroup: ViewGroup):
        if viewGroup == ViewGroup.SCENE:
            return self._getDefaultViewsXYZ()
        elif viewGroup == ViewGroup.SOLIDS:
            views = []
            for solid in self._scene.solids:
                views += self._getDefaultSolidViewsXYZ(solid)
            return views
        else:
            raise NotImplementedError("Default views for ViewGroup {} not implemented.".format(viewGroup))

    def _getDefaultSolidViewsXYZ(self, solid: Solid) -> List[View2D]:
        raise NotImplementedError()

    def _buildViews(self, views: List[View2D]):
        raise NotImplementedError()

    @staticmethod
    def _getDefaultViewsXYZ() -> list:
        return [View2D(Direction.X_POS, Direction.Z_POS),
                View2D(Direction.Y_NEG, Direction.Z_POS),
                View2D(Direction.Z_POS, Direction.X_NEG)]


# todo: this scene dependence is not ideal ...
# init:
    # need bbox of scene
    # bbox of each solid
    # surface labels of each solid
    # todo: 1 or 3 views per surface?
# _buildGroup (or build):
    # todo: set View name with view group name and other related info?
    #  Better, only set view group. Then create getName inside View2D which is dynamic and include current projection direction, view type, viewGroup, interaction keys
