import env

TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import loadSolid, MayaviViewer

    solid = loadSolid("pytissueoptics/examples/scene/droid.obj")

    viewer = MayaviViewer()
    viewer.add(solid, representation="surface", showNormals=True, normalLength=0.2)
    viewer.show()


if __name__ == "__main__":
    exampleCode()
