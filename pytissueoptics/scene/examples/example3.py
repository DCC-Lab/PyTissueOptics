TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import loadSolid, MayaviViewer

    solid = loadSolid("exampleFile.obj")

    viewer = MayaviViewer()
    viewer.add(solid, representation="surface", showNormals=True)
    viewer.show()


if __name__ == "__main__":
    exampleCode()
