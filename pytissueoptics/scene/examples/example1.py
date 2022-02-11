TITLE = "Transforms on a Solid"

DESCRIPTION = """  Translation Transform and Rotation Transform can be applied on Solids.
Rotation parameters are in degrees and represent the angle of rotation around each cartesian axis.
Here a cube is rotated, then copied, then the copied cube is translated, but will keep its rotation."""



def exampleCode():
    from pytissueoptics.scene import Vector, Cuboid, MayaviViewer
    from copy import deepcopy

    cube = Cuboid(1, 1, 1, position=Vector(1, 1, 10))
    #cube.rotate(xTheta=45, yTheta=45, zTheta=0)
    cube2 = deepcopy(cube)
    cube2.translateTo(Vector(0, 0, 0))

    viewer = MayaviViewer()
    viewer.add(cube, representation="surface")
    viewer.add(cube2, representation="surface")
    viewer.show()


if __name__ == "__main__":
    exampleCode()
