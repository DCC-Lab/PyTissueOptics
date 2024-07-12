import re

import numpy as np

from pytissueoptics import *
import matplotlib.pyplot as plt


def run(x) -> float:
    N = 100000

    tissue = ScatteringScene([Cube(60, material=ScatteringMaterial(mu_a=x, mu_s=1, g=0.01, n=1))])  #  mu_s=1, g=0.89, n=1.37))])
    logger = EnergyLogger(tissue, defaultBinSize=0.1)
    source = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)

    source.propagate(tissue, logger=logger, showProgress=True)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats(saveToFile="tmp.txt", verbose=False)
    with open("tmp.txt", "r") as f:
        stats = f.read()
    match = re.search(r"Absorbance: (\d+\.\d+)%", stats)
    absorbance = float(match.group(1))
    return absorbance


mu_a_values = np.linspace(0.001, 0.05, 10, dtype=float)
absorbance_values = []
for i, mu_a in enumerate(mu_a_values):
    print(f"Running simulation {i+1}/{len(mu_a_values)}")
    absorbance = run(mu_a)
    absorbance_values.append(absorbance)

plt.plot(mu_a_values, absorbance_values)
plt.xlabel("Absorption coefficient (1/mm)")
plt.ylabel("Absorbance (%)")
plt.show()

# With mua from 0.001 to 0.05, the absorbance increases from 6% to 46%.
