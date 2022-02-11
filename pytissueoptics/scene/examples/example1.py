TITLE = "Rotating a Solid"

DESCRIPTION = """  Spawning a Cuboid() of side lenghts = 1,1,1 @(0,0,1) and applying a rotation transformation.
Rotation parameters are in degrees and represent the angle of rotation around each cartesian axis."""


def exampleCode():
    from pytissueoptics.scene import Vector, Cuboid, MayaviViewer
    cube = Cuboid(1, 1, 1, position=Vector(0, 0, 1))
    cube.rotate(xTheta=45, yTheta=45, zTheta=0)

    viewer = MayaviViewer()
    viewer.add(cube, representation="surface")
    viewer.show()


if __name__ == "__main__":
    exampleCode()
