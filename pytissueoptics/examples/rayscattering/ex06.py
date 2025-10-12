import env  # noqa: F401

from pytissueoptics import *  # noqa: F403

TITLE = "Detectors"

DESCRIPTION = """ Sampling volume simulation with a directional source and a circle detector. """


def exampleCode():
    import numpy as np

    N = 1000000 if hardwareAccelerationIsAvailable() else 1000

    material = ScatteringMaterial(mu_s=10, mu_a=1, g=0.98, n=1.0)

    cube = Cuboid(a=3, b=3, c=3, position=Vector(0, 0, 0), material=material, label="cube")

    detectorA = Circle(
        radius=0.25,
        orientation=Vector(0, 0, -1),
        position=Vector(0, 1, 1.501),
        label="detectorA",
    ).asDetector(halfAngle=np.pi / 4)

    detectorB = Circle(
        radius=0.25,
        orientation=Vector(0, 0, -1),
        position=Vector(0, -1, 1.501),
        label="detectorB",
    ).asDetector(halfAngle=np.pi / 4)

    scene = ScatteringScene([cube, detectorA, detectorB])

    logger = EnergyLogger(scene)
    source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), N=N, diameter=0.5)

    source.propagate(scene, logger)

    viewer = Viewer(scene, source, logger)
    viewer.reportStats()

    # Either filter the whole logger
    logger.filter(detectedBy=["detectorA", "detectorB"])
    viewer.show2D(View2DProjectionX())
    viewer.show3D()

    # ... or keep the logger intact and filter individual views
    # viewer.show2D(View2DProjectionX(detectedBy="detectorA"))
    # viewer.show3D(pointCloudStyle=PointCloudStyle(detectedBy=["detectorA", "detectorB"]))


if __name__ == "__main__":
    exampleCode()
