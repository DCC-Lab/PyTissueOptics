import env  # noqa: F401

TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import get3DViewer, loadSolid

    solid = loadSolid("pytissueoptics/examples/scene/droid.obj")

    viewer = get3DViewer()
    viewer.add(solid, representation="surface", showNormals=True, normalLength=0.2)
    viewer.show()


if __name__ == "__main__":
    exampleCode()
