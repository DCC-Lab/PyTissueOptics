TITLE = "Explore different Shapes"

DESCRIPTION = """  
Spawning a Cuboid() of side lengths = 1,3,1 at (1,0,0)
Sphere() of radius 0.5 at (0,0,0).
Ellipsoid() of eccentricity 3,1,1 at (-2, 0, 0)
A Mayavi Viewer allows for the display of those solids."""


def exampleCode():
    from pytissueoptics.scene import Vector, Cuboid, Sphere, Ellipsoid, MayaviViewer

    cuboid = Cuboid(a=1, b=3, c=1, position=Vector(1, 0, 0))
    sphere = Sphere(radius=0.5, position=Vector(0, 0, 0))
    ellipsoid = Ellipsoid(a=1.5, b=1, c=1, position=Vector(-2, 0, 0))

    viewer = MayaviViewer()
    viewer.add(cuboid, sphere, ellipsoid, representation="surface")
    viewer.show()


if __name__ == "__main__":
    import env
    exampleCode()
