TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import SolidModel, MayaviViewer

    solid = SolidModel("exampleFile.obj")

    viewer = MayaviViewer()
    viewer.add(*solid.getSolids(), representation="surface")
    viewer.show()


if __name__ == "__main__":
    exampleCode()
