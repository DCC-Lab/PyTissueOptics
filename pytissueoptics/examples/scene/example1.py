import env  # noqa: F401

TITLE = "Transforms on a Solid"

DESCRIPTION = """  Translation Transform and Rotation Transform can be applied on Solids.
Rotation parameters are in degrees and represent the angle of rotation around each cartesian axis.
Here a cube is translated, another is rotated."""


def exampleCode():
    from pytissueoptics.scene import Cuboid, Vector, get3DViewer

    centerCube = Cuboid(a=1, b=1, c=1, position=Vector(0, 0, 0))
    topCube = Cuboid(a=1, b=1, c=1, position=Vector(0, 2, 0))
    bottomCube = Cuboid(a=1, b=1, c=1, position=Vector(0, 0, 0))

    bottomCube.translateTo(Vector(0, -2, 0))
    centerCube.rotate(0, 30, 30)

    viewer = get3DViewer()
    viewer.add(centerCube, topCube, bottomCube, representation="surface")
    viewer.show()


if __name__ == "__main__":
    exampleCode()
