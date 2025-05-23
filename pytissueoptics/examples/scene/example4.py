import env  # noqa: F401

TITLE = "Lenses"

DESCRIPTION = """Explore different types of lens-shaped solids."""


def exampleCode():
    from pytissueoptics.scene import (
        PlanoConcaveLens,
        PlanoConvexLens,
        RefractiveMaterial,
        SymmetricLens,
        ThickLens,
        Vector,
        get3DViewer,
    )

    material = RefractiveMaterial(refractiveIndex=1.44)
    lens1 = ThickLens(30, 60, diameter=25.4, thickness=4, material=material, position=Vector(0, 0, 0))
    lens3 = SymmetricLens(f=60, diameter=25.4, thickness=4, material=material, position=Vector(0, 0, 10))
    lens2 = PlanoConvexLens(f=-60, diameter=25.4, thickness=4, material=material, position=Vector(0, 0, 20))
    lens4 = PlanoConcaveLens(f=60, diameter=25.4, thickness=4, material=material, position=Vector(0, 0, 30))

    viewer = get3DViewer()
    viewer.add(lens1, lens2, lens3, lens4, representation="surface", colormap="viridis", showNormals=False)
    viewer.show()


if __name__ == "__main__":
    exampleCode()
