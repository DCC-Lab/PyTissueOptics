"""Quick performance benchmark: CPU (pure-Python) vs OpenCL (hardware) Monte Carlo.

Propagates a collimated pencil beam through a simple infinite scattering medium
on each backend, then reports throughput (photons/s) and speedup. As a physics
sanity check it also reports total absorbance (energy conservation) and the
energy-weighted mean penetration depth, which should agree within Monte Carlo
noise between the two backends.

Run:  python benchmark_cpu_vs_opencl.py [N_cpu] [N_gpu]
"""

import sys
import time
from dataclasses import dataclass

import numpy as np

from pytissueoptics import (
    EnergyLogger,
    PencilPointSource,
    ScatteringMaterial,
    Stats,
    Vector,
    hardwareAccelerationIsAvailable,
    samples,
)
from pytissueoptics.rayscattering.opencl.CLScene import WORLD_SOLID_LABEL
from pytissueoptics.scene.logger.logger import InteractionKey

# A simple, single-material scattering environment. Moderate absorption keeps the
# pure-Python CPU run quick by terminating photons in a reasonable number of steps.
MATERIAL = ScatteringMaterial(mu_s=30.0, mu_a=2.0, g=0.9)
SEED = 1


@dataclass
class Result:
    backend: str
    N: int
    seconds: float
    absorbance: float  # % of input energy absorbed in the medium
    meanDepth: float  # energy-weighted mean penetration depth along z

    @property
    def throughput(self) -> float:
        return self.N / self.seconds


def runOnce(useHardware: bool, N: int) -> Result:
    tissue = samples.InfiniteTissue(MATERIAL)
    logger = EnergyLogger(tissue)
    source = PencilPointSource(
        position=Vector(0, 0, 0),
        direction=Vector(0, 0, 1),
        N=N,
        useHardwareAcceleration=useHardware,
        seed=SEED,
    )

    start = time.perf_counter()
    source.propagate(tissue, logger=logger, showProgress=False)
    seconds = time.perf_counter() - start

    stats = Stats(logger)
    absorbance = stats.getAbsorbance(WORLD_SOLID_LABEL, useTotalEnergy=True)

    points = logger.getDataPoints(InteractionKey(WORLD_SOLID_LABEL, None))
    weight, z = points[:, 0], points[:, 3]
    meanDepth = float(np.sum(weight * z) / np.sum(weight))

    backend = "OpenCL" if useHardware else "CPU"
    return Result(backend, N, seconds, absorbance, meanDepth)


def printResult(r: Result) -> None:
    print(
        f"  {r.backend:7s}  N={r.N:>8d}  {r.seconds:8.3f} s  "
        f"{r.throughput:12,.0f} photons/s  "
        f"absorbance={r.absorbance:7.3f}%  meanDepth={r.meanDepth:.4f}"
    )


def main() -> None:
    nCPU = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    nGPU = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000

    print(f"Material: mu_s={MATERIAL.mu_s}, mu_a={MATERIAL.mu_a}, g={MATERIAL.g}  (infinite tissue)")
    print(f"OpenCL available: {hardwareAccelerationIsAvailable()}\n")

    cpu = runOnce(useHardware=False, N=nCPU)
    printResult(cpu)

    if not hardwareAccelerationIsAvailable():
        print("\nOpenCL is not available on this machine; skipping the hardware run.")
        return

    gpu = runOnce(useHardware=True, N=nGPU)
    printResult(gpu)

    speedup = gpu.throughput / cpu.throughput
    depthDiff = abs(gpu.meanDepth - cpu.meanDepth)
    print(f"\nSpeedup (throughput): {speedup:,.1f}x faster on OpenCL")
    print(f"Mean-depth agreement: |OpenCL - CPU| = {depthDiff:.4f} "
          f"({100 * depthDiff / cpu.meanDepth:.2f}% of CPU value)")


if __name__ == "__main__":
    main()
