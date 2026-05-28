"""
Figure 1b – Pencil beam through a three-layer stack with scattering (PhantomTissue).

Same geometry and incidence angle as Figure 1, but using the PhantomTissue sample
(μs=[2, 3, 2], μa=[1, 1, 2], g=0.8, n=[1.4, 1.7, 1.4]) to show the effect of
scattering compared to the pure refraction case.
"""

import math

from pytissueoptics import *  # noqa: F403

TITLE = "Pencil beam through a three-layer stack with scattering (PhantomTissue)"

DESCRIPTION = """ Propagation of a pencil beam through the PhantomTissue sample with scattering enabled.
Comparison to Figure 1 (μs=0) to illustrate the effect of scattering on beam propagation. """


def exampleCode():
    N = 100000 if hardwareAccelerationIsAvailable() else 1000

    tissue = samples.PhantomTissue()
    logger = EnergyLogger(tissue)

    angle_deg = 30
    angle_rad = math.radians(angle_deg)
    direction = Vector(math.sin(angle_rad), 0, math.cos(angle_rad))

    source = PencilPointSource(
        position=Vector(-0.5, 0, -0.3), direction=direction, N=N, displaySize=0.3
    )

    tissue.show(source=source)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionX(energyType=EnergyType.FLUENCE_RATE))
    viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
    viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface1", surfaceEnergyLeaving=False))
    viewer.show1D(Direction.Z_POS)
    viewer.show3D()
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))


if __name__ == "__main__":
    exampleCode()
