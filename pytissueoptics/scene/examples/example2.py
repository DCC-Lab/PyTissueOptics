TITLE = "Stacking Cuboids"

DESCRIPTION = """  It is possible to stack multiple cuboids toghether, which will manage the interface materials.
TO be stackable in a particular axis, the cuboids must have the same size in that axis."""


def exampleCode():
    from pytissueoptics.scene import Cuboid, MayaviViewer
    cuboid1 = Cuboid(1, 1, 1)
    cuboid2 = Cuboid(2, 1, 1)
    cuboidStack = cuboid1.stack(cuboid2, onSurface="Right")

    viewer = MayaviViewer()
    viewer.add(cuboidStack, representation="wireframe", lineWidth=5)
    viewer.show()


if __name__ == "__main__":
    exampleCode() 
