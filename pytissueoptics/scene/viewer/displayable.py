from abc import abstractmethod

from pytissueoptics.scene.viewer.mayavi import MayaviViewer


class Displayable:
    @abstractmethod
    def addToViewer(self, viewer, **kwargs):
        pass

    def show(self, **kwargs):
        viewer = MayaviViewer()
        self.addToViewer(viewer, **kwargs)
        viewer.show()
