import time

import matplotlib.pyplot as plt
import numpy as np

from pytissueoptics import *
from pytissueoptics.rayscattering.opencl import CONFIG


def computeOptimalNWorkUnits() -> int:
    """
    Find the optimal amount of work units for the OpenCL kernel (hardware-specific).
    """

    USE_CONSTANT_N = False  # uses N_WORK_UNITS * 5 if false, else N below

    N = 50000
    AVERAGING = 3

    material1 = ScatteringMaterial(mu_s=5, mu_a=0.8, g=0.9, n=1.4)
    material2 = ScatteringMaterial(mu_s=10, mu_a=0.8, g=0.9, n=1.7)
    cube = Cuboid(a=3, b=3, c=3, position=Vector(0, 0, 0), material=material1, label="Cube")
    sphere = Sphere(radius=1, order=2, position=Vector(0, 0, 0), material=material2, label="Sphere",
                    smooth=True)
    scene = RayScatteringScene([cube, sphere])

    arr_workUnits, arr_speed = [], []
    for i in range(14, 30+1):
        CONFIG.N_WORK_UNITS = int(np.sqrt(2) ** i)

        if not USE_CONSTANT_N:
            N = CONFIG.N_WORK_UNITS * 5

        timePerPhoton = 0
        totalTime = 0
        for _ in range(AVERAGING):
            source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), N=N,
                                       useHardwareAcceleration=True, diameter=0.5)

            logger = Logger()
            t0 = time.time()
            source.propagate(scene, logger=logger, showProgress=False)
            elapsedTime = time.time() - t0
            totalTime += elapsedTime

            timePerPhoton += elapsedTime / N
        timePerPhoton /= AVERAGING
        totalTime /= AVERAGING
        print(f"{CONFIG.N_WORK_UNITS} \t work units : {timePerPhoton:.6f} s/p [{AVERAGING}x {totalTime:.2f}s]")

        arr_workUnits.append(CONFIG.N_WORK_UNITS)
        arr_speed.append(timePerPhoton * 10**6)

    CONFIG.N_WORK_UNITS = None
    CONFIG.save()

    plt.plot(arr_workUnits, arr_speed, 'o')
    plt.xlabel("N_WORK_UNITS")
    plt.ylabel("Time per photon (us)")
    plt.semilogy()

    print(f"Found an optimal N_WORK_UNITS of {arr_workUnits[np.argmin(arr_speed)]}. \nPlease analyze and close the "
          f"plot to continue.")
    plt.show()

    return arr_workUnits[np.argmin(arr_speed)]


if __name__ == "__main__":
    CONFIG.AUTO_SAVE = False
    optimalNWorkUnits = computeOptimalNWorkUnits()
