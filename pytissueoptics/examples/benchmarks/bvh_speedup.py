"""
Benchmark: GPU BVH path vs. legacy flat per-solid AABB path.

Measures end-to-end propagate() time on three sphere-rich scenes, with both BVH disabled and
enabled. Run with hardware acceleration available; otherwise the script will report it cannot
benchmark and exit.

Usage:
    python -m pytissueoptics.examples.benchmarks.bvh_speedup
"""

from __future__ import annotations

import time
from typing import Callable, Tuple

import numpy as np

import env  # noqa: F401  -- adds the package root to sys.path

import pytissueoptics.rayscattering.opencl.CLPhotons as cl_photons
from pytissueoptics import (
    Cube,
    Cuboid,
    EnergyLogger,
    PencilPointSource,
    ScatteringMaterial,
    ScatteringScene,
    Sphere,
    Vector,
    hardwareAccelerationIsAvailable,
)
from pytissueoptics.rayscattering.opencl.CLScene import CLScene


def _phantomShells() -> ScatteringScene:
    outer = Sphere(2.5, order=2, material=ScatteringMaterial(mu_a=0.04, mu_s=0.09, g=0.89, n=1.37), label="outer")
    inner = Sphere(2.3, order=2, material=ScatteringMaterial(mu_a=0.2, mu_s=90, g=0.89, n=1.37), label="inner")
    core = Sphere(1.0, order=2, material=ScatteringMaterial(mu_a=0.5, mu_s=1e-6, g=1, n=1.37), label="core")
    grid = Cuboid(6, 6, 6, material=ScatteringMaterial(mu_a=0.2, mu_s=70, g=0.89, n=1.37))
    return ScatteringScene([core, inner, outer, grid])


def _cubeWithSphere() -> ScatteringScene:
    cube = Cube(60, material=ScatteringMaterial(mu_a=0.005, mu_s=0, g=0.01, n=1.37))
    sphere = Sphere(15, order=3, material=ScatteringMaterial(mu_a=0.002, mu_s=5, g=0.9, n=1))
    return ScatteringScene([cube, sphere])


def _denseSpheres() -> ScatteringScene:
    """Two finely-tessellated spheres - lots of polygons, few solids - to stress the BVH."""
    materialA = ScatteringMaterial(mu_a=0.1, mu_s=10, g=0.85, n=1.37)
    materialB = ScatteringMaterial(mu_a=0.4, mu_s=3, g=0.85, n=1.37)
    big = Sphere(2.0, order=3, material=materialA, label="big")
    small = Sphere(0.6, order=3, position=Vector(0.5, 0, 0), material=materialB, label="small")
    return ScatteringScene([big, small])


def _runOnce(buildScene: Callable[[], ScatteringScene], N: int, useBVH: bool) -> Tuple[float, int, int, int]:
    scene = buildScene()
    np.random.seed(0)
    src = PencilPointSource(position=Vector(0, 0.01, -3), direction=Vector(0, 0, 1), N=N, displaySize=1)
    if isinstance(scene.solids[0], Cube) or scene.getBoundingBox().xWidth > 50:
        src = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)
    logger = EnergyLogger(scene, defaultBinSize=0.5, keep3D=False)

    inspectScene = CLScene(scene, nWorkUnits=1024, useBVH=useBVH)
    nSolids = int(inspectScene.nSolids)
    nTriangles = len(inspectScene.triangles._trianglesInfo)
    nNodes = int(inspectScene.nNodes)

    orig = cl_photons.CLScene
    cl_photons.CLScene = lambda s, nWU: orig(s, nWU, useBVH=useBVH)
    try:
        t0 = time.time()
        src.propagate(scene, logger=logger, showProgress=False)
        elapsed = time.time() - t0
    finally:
        cl_photons.CLScene = orig
    return elapsed, nSolids, nTriangles, nNodes


def _bench(name: str, buildScene: Callable[[], ScatteringScene], N: int) -> None:
    print(f"\n=== {name}  (N={N}) ===")
    # One warmup run per path so kernel build / IPP estimate don't pollute the timing.
    _runOnce(buildScene, N=max(500, N // 20), useBVH=False)
    _runOnce(buildScene, N=max(500, N // 20), useBVH=True)

    tFlat, nSolids, nTriangles, _ = _runOnce(buildScene, N=N, useBVH=False)
    tBVH, _, _, nNodes = _runOnce(buildScene, N=N, useBVH=True)
    speedup = tFlat / tBVH if tBVH > 0 else float("inf")
    print(f"  scene: {nSolids} solids, {nTriangles} triangles, BVH nodes: {nNodes}")
    print(f"  flat: {tFlat:7.3f} s    BVH: {tBVH:7.3f} s    speedup: {speedup:5.2f}x")


def exampleCode():
    if not hardwareAccelerationIsAvailable():
        print("Hardware acceleration is not available; the BVH benchmark requires OpenCL.")
        return

    _bench("Spherical shells (4 solids, ~970 tris)", _phantomShells, N=20_000)
    _bench("Cube with sphere inclusion (2 solids, ~1300 tris)", _cubeWithSphere, N=20_000)
    _bench("Two dense spheres (2 solids, ~2560 tris)", _denseSpheres, N=20_000)


TITLE = "BVH speedup vs flat per-solid AABB"
DESCRIPTION = "Compares end-to-end propagation time with the BVH disabled and enabled."


if __name__ == "__main__":
    exampleCode()
