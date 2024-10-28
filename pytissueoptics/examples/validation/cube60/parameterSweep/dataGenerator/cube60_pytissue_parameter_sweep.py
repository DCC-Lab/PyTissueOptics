from typing import List
import numpy as np
from pytissueoptics import *
from utils import SweepSimResult, g_values, mu_a_values, mu_s_values


def pytissueoptics_cube60_sweep_simulator(mu_a_values: List[float], mu_s_values: List[float], g_values: List[float], N: int = 100000, seed: int = 0) -> List[SweepSimResult]:
    N = N if hardwareAccelerationIsAvailable() else 10000
    np.random.seed(seed)
    results = []
    for mu_a in mu_a_values:
        for mu_s in mu_s_values:
            for g in g_values:
                tissue = ScatteringScene([Cube(60, material=ScatteringMaterial(mu_a=mu_a, mu_s=mu_s, g=g, n=1), label="cube")])
                logger = EnergyLogger(tissue, defaultBinSize=0.1)
                source = PencilPointSource(position=Vector(0, 0, -30), direction=Vector(0, 0, 1), N=N, displaySize=1)
                source.propagate(tissue, logger=logger)
                stats = Stats(logger)
                results.append(SweepSimResult(g=g, mus=mu_s, absorbance=stats.getAbsorbance("cube"), mua=mu_a))
                print(f"Results for (mu_a={mu_a}, mus={mu_s},g={g}):: absorbance={results[-1].absorbance}")

    return results


if __name__ == "__main__":
    results = pytissueoptics_cube60_sweep_simulator([0.005], [30.0], [0.95], N=100000, seed=2)
    SweepSimResult.save_to_json(results, "../cube60_sweep_results.json", software="pytissueoptics")
