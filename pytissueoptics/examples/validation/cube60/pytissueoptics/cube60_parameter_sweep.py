import json
from dataclasses import dataclass, asdict

import numpy as np
import seaborn as sns
from pytissueoptics import *
import matplotlib.pyplot as plt

TITLE = "CUBE60 Validation"
DESCRIPTION = """Validation of the cube60 benchmark from mcx. Comparison of the total energy absorbed in PyTissueOptics,
 MCX from QFang and mcxyz from Prahl et al. 2006."""

N = 100000 if hardwareAccelerationIsAvailable() else 10000


@dataclass
class SweepSimResult:
    g: float
    mus: float
    absorbance: float
    mua: float


def cube60_sweep_g_mus(seed=0):
    np.random.seed(seed)
    mua = 0.03
    results = []
    mus_values = [0.005, 0.01, 0.015, 0.020, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.060, 0.065, 0.07, 0.075,
                  0.08, 0.085, 0.09, 0.095, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.7,
                  0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 7, 10, 15, 20, 25, 30]
    g_values = [1, 0.999, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 0.3, 0]
    for mus in mus_values:
        for g in g_values:
            tissue = ScatteringScene([Cube(60, material=ScatteringMaterial(mu_a=mua, mu_s=mus, g=g, n=1), label="cube")])
            logger = EnergyLogger(tissue, defaultBinSize=0.1)
            source = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)
            source.propagate(tissue, logger=logger)
            stats = Stats(logger)
            results.append(SweepSimResult(g=g, mus=mus, absorbance=stats.getAbsorbance("cube"), mua=mua))
            print(f"Finished simulation for g={g}, mus={mus}, absorbance={results[-1].absorbance}")

    return results


def save_results_to_json(results, filename):
    with open(filename, 'w') as f:
        json.dump([asdict(result) for result in results], f, indent=4)

def load_results_from_json(filename):
    with open(filename, 'r') as f:
        results_dicts = json.load(f)
    return [SweepSimResult(**result) for result in results_dicts]


def plot_absorbance(results, g_value=None, mus_value=None):
    x = []
    y = []
    for result in results:
        if g_value is not None and result.g == g_value:
            x.append(result.mus)
            y.append(result.absorbance)
        elif mus_value is not None and result.mus == mus_value:
            x.append(result.g)
            y.append(result.absorbance)

    plt.plot(x, y, 'o-')
    plt.xlabel('g' if g_value is None else 'mus')
    plt.ylabel('Total Energy Absorbed (%)')
    plt.show()

def plot_absorbance_difference(a, b, g_value=None, mus_value=None):
    x = []
    y = []
    for result_a in a:
        if g_value is not None and result_a.g == g_value:
            x.append(result_a.mus)
            y.append(result_a.absorbance - result_b.absorbance)
        elif mus_value is not None and result_a.mus == mus_value:
            x.append(result_a.g)
            y.append(result_a.absorbance - result_b.absorbance)

    plt.plot(x, y, 'o-')
    plt.xlabel('g' if g_value is None else 'mus')
    plt.ylabel(r'Total Energy Absorbed Difference ($\delta \%$)')
    plt.show()


if __name__ == "__main__":
    #results = cube60_sweep_g_mus()
    #save_results_to_json(results, 'cube60_sweep_results_1.json')
    resultsPyTissueOptics = load_results_from_json('cube60_sweep_results_1.json')
    resultsMCX = load_results_from_json('cube60_sweep_mcx_results.json')

    plot_absorbance_difference(resultsPyTissueOptics, resultsMCX, g_value=0.8)
