import time

import matplotlib.pyplot as plt
import numpy as np

from pytissueoptics import *
from pytissueoptics.rayscattering.opencl import CLParameters as clp

"""
Find the optimal amount of work units for the OpenCL kernel (hardware-specific).
"""

# fixme: the results are not very consistent.
#  It is hard to conclude an optimal amount of work units for my hardware at least...
#  from experimentation it should be around 10-20k, but it varies between 8k and 65k

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
for i in range(14, 31+1):
    clp.N_WORK_UNITS = int(np.sqrt(2) ** i)

    if not USE_CONSTANT_N:
        N = clp.N_WORK_UNITS * 5

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
    print(f"{clp.N_WORK_UNITS} \t work units : {timePerPhoton:.6f} s/p [{AVERAGING}x {totalTime:.2f}s]")

    arr_workUnits.append(clp.N_WORK_UNITS)
    arr_speed.append(timePerPhoton * 10**6)

plt.plot(arr_workUnits, arr_speed, 'o')
plt.xlabel("N_WORK_UNITS")
plt.ylabel("Time per photon (us)")
plt.semilogy()

print(f"Optimal N_WORK_UNITS tested: {arr_workUnits[np.argmin(arr_speed)]}")

plt.show()
