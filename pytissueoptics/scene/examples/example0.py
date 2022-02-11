TITLE = "A Cube and a Sphere"

DESCRIPTION = """  Spawning a Cuboid() of side lenghts = 2,3,1 @(0,0,1) and a Sphere() of radius 0.1 @(0,0,0).
 A Mayavi Viewer allows for the display of those solids."""


def exampleCode():
    from pytissueoptics.scene import Vector, Cuboid, Sphere, MayaviViewer
    cuboid = Cuboid(2, 3, 1, position=Vector(1, 0, 0))
    sphere = Sphere(radius=0.1, position=Vector(0, 0, 0))

    viewer = MayaviViewer()
    viewer.add(cuboid, sphere, representation="surface")
    viewer.show()


if __name__ == "__main__":
    exampleCode()
