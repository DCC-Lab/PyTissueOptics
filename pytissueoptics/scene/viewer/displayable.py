from abc import abstractmethod

from .provider import get3DViewer


class Displayable:
    @abstractmethod
    def addToViewer(self, viewer, **kwargs):
        pass

    def show(self, **kwargs):
        viewer = get3DViewer()
        self.addToViewer(viewer, **kwargs)
        viewer.show()
