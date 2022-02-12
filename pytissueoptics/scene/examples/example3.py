TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import Loader, MayaviViewer

    loader = Loader()
    scene = loader.load("exampleFile.obj")

    viewer = MayaviViewer()
    viewer.add(*scene, representation="surface")
    viewer.show()


if __name__ == "__main__":
    exampleCode()
