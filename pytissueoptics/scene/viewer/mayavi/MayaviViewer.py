try:
    from mayavi import mlab
except ImportError:
    pass

from pytissueoptics.scene.viewer.mayavi.MayaviSolid import MayaviSolid
from pytissueoptics.scene.solids.sphere import Sphere


class MayaviViewer:
    def __init__(self):
        mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(400, 300))
        mlab.clf()

    def addMesh(self, other: 'MayaviSolid', representation="wireframe", line_width=0.25):
        x,y,z,tri = other.meshComponents
        mlab.triangular_mesh(x,y,z,tri, representation=representation, line_width=line_width)

    def show(self):
        mlab.view(132, 54, 45, [21, 20, 21.5])
        mlab.show()


if __name__ == "__main__":

    from mayavi import mlab
    mayaviObject = MayaviSolid(Sphere())
    viewer = MayaviViewer()
    viewer.addMesh(mayaviObject)
    viewer.show()
