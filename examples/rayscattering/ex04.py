from pytissueoptics import *

TITLE = "Propagation in a custom layer stack."

DESCRIPTION = """  
This example shows how to make a layer stack. You initialize multiple Cuboid() and stack them in a layer stack.
You can do this using cuboid.stack(secondCuboid). The cuboid sides on the stacked face has to have the same dimensions
on both cuboids.
"""


def exampleCode():
    materialLayer1 = ScatteringMaterial(mu_s=2, mu_a=0.5, g=0.7, n=1.3)
    materialLayer2 = ScatteringMaterial(mu_s=5, mu_a=0.8, g=0.8, n=1.4)
    materialLayer3 = ScatteringMaterial(mu_s=50, mu_a=2.5, g=0.9, n=1.5)

    layer1 = Cuboid(a=10, b=10, c=1, position=Vector(0, 0, 0), material=materialLayer1, label="Layer 1")
    layer2 = Cuboid(a=10, b=10, c=1, position=Vector(0, 0, 0), material=materialLayer2, label="Layer 2")
    layer3 = Cuboid(a=10, b=10, c=1, position=Vector(0, 0, 0), material=materialLayer3, label="Layer 3")
    stack1 = layer1.stack(layer2, "back")
    stackedTissue = stack1.stack(layer3, "back")

    tissue = ScatteringScene([stackedTissue])

    logger = EnergyLogger(tissue)
    source = PencilPointSource(position=Vector(0, 0, -5), direction=Vector(0, 0, 1), N=500)
    source.propagate(tissue, logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()
    viewer.show3D()


if __name__ == "__main__":
    exampleCode()
