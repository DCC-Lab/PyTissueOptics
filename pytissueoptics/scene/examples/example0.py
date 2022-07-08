TITLE = "Explore different Shapes"

DESCRIPTION = """  
Spawning a Cuboid() of side lengths = 1,3,1 at (1,0,0)
Sphere() of radius 0.5 at (0,0,0).
Ellipsoid() of eccentricity 3,1,1 at (-2, 0, 0)
A Mayavi Viewer allows for the display of those solids."""


def exampleCode():
    from pytissueoptics.scene import Vector, Cuboid, Sphere, Cone, MayaviViewer
    import numpy as np
    cuboid = Cuboid(a=1, b=1, c=1, position=Vector(0, 0, 0))
    cuboid2 = Cuboid(a=1, b=1, c=1, position=Vector(1.01, 0, 0))
    cuboid3 = Cuboid(a=1, b=2, c=1)
    stack1 = cuboid.stack(cuboid2, "top")
    stack2 = stack1.stack(cuboid3, "front")
    # sphere = Sphere(radius=0.5, position=Vector(0, 0, 0))
    # cylinder = Cone(height=3)
    # cylinder2 = Cone(height=3)
    # ellipsoid = Ellipsoid(a=1.5, b=0.5, c=1, position=Vector(-2, 0, 0), order=4)

    viewer = MayaviViewer()
    s1 = viewer.add(stack2, representation="surface", constantColor=False, color=(0.8125, 0.8359, 0.89844))[0]
    scene = s1.parent.parent.parent.parent
    scene.scene.background = (1, 1, 1)
    s1.actor.property.color = (0.8125, 0.8359, 0.89844)
    s1.actor.property.edge_visibility = True
    s1.actor.property.line_width = 2.5
    s1.actor.property.edge_color = (25/255, 152/255, 158/255)
    s1.actor.property.interpolation = "flat"

    # s2.actor.property.color = (0.8125, 0.8359, 0.89844)
    # s2.actor.property.edge_visibility = True
    # s2.actor.property.line_width = 2.5
    # s2.actor.property.edge_color = (238/255, 51/255, 238/255)
    # s2.actor.property.interpolation = "flat"

    viewer.show()


if __name__ == "__main__":
    exampleCode()
