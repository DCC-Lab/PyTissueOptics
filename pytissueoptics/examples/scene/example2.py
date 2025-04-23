import env  # noqa: F401

TITLE = "Stacking Cuboids"

DESCRIPTION = """  It is possible to stack multiple cuboids together, which will manage the interface materials.
To be stackable in a particular axis, the cuboids must have the same size in that axis."""


def exampleCode():
    from pytissueoptics.scene import Cuboid, Vector, get3DViewer

    cuboid1 = Cuboid(1, 1, 1, position=Vector(2, 0, 0))
    cuboid2 = Cuboid(2, 1, 1, position=Vector(0, 2, 0))
    cuboid3 = Cuboid(3, 1, 1, position=Vector(0, 0, 2))

    viewer = get3DViewer()
    viewer.add(cuboid1, cuboid2, cuboid3, representation="wireframe", lineWidth=5)
    viewer.show()

    cuboidStack = cuboid1.stack(cuboid2, onSurface="right")
    cuboidStack = cuboidStack.stack(cuboid3, onSurface="top")

    viewer = get3DViewer()
    viewer.add(cuboidStack, representation="wireframe", lineWidth=5)
    viewer.show()


if __name__ == "__main__":
    exampleCode()
