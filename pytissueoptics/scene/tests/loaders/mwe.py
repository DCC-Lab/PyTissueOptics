from pytissueoptics.scene.loaders import Loader
from pytissueoptics.scene.viewer.mayavi import MayaviSolid, MayaviViewer

loader = Loader()
solidObjects = loader.load("./parsers/objFiles/brainModel.obj")
viewer = MayaviViewer()
for obj in solidObjects:
    viewer.addMayaviSolid(MayaviSolid(obj), representation="surface")


viewer.show()