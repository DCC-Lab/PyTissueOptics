"""
Benchmark: GPU BVH path vs. legacy flat per-solid AABB path.

Measures propagate() time on three sphere-rich scenes, with both BVH disabled and enabled,
and breaks the total down into BVH/scene construction time vs. propagation kernel time. The
construction cost is non-trivial for small scenes and is what a user actually pays in
end-to-end runtime, so the report includes both speedup figures (propagation-only and total).

Run with hardware acceleration available; otherwise the script will report it cannot
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


def _runOnce(buildScene: Callable[[], ScatteringScene], N: int, useBVH: bool) -> Tuple[float, float, int, int, int]:
    scene = buildScene()
    np.random.seed(0)
    src = PencilPointSource(position=Vector(0, 0.01, -3), direction=Vector(0, 0, 1), N=N, displaySize=1)
    if isinstance(scene.solids[0], Cube) or scene.getBoundingBox().xWidth > 50:
        src = PencilPointSource(position=Vector(0, 0, -29.99), direction=Vector(0, 0, 1), N=N, displaySize=1)
    logger = EnergyLogger(scene, defaultBinSize=0.5, keep3D=False)

    # Measure scene-build cost (which includes BVH construction when useBVH=True) in isolation.
    # propagate() builds its own CLScene internally, so this same cost is also paid inside the
    # total timer below; subtracting gives an approximate kernel-only time.
    tBuildStart = time.time()
    inspectScene = CLScene(scene, nWorkUnits=1024, useBVH=useBVH)
    tBuild = time.time() - tBuildStart
    nSolids = int(inspectScene.nSolids)
    nTriangles = len(inspectScene.triangles._trianglesInfo)
    nNodes = int(inspectScene.nNodes)

    orig = cl_photons.CLScene
    cl_photons.CLScene = lambda s, nWU: orig(s, nWU, useBVH=useBVH)
    try:
        t0 = time.time()
        src.propagate(scene, logger=logger, showProgress=False)
        tTotal = time.time() - t0
    finally:
        cl_photons.CLScene = orig
    return tTotal, tBuild, nSolids, nTriangles, nNodes


def _bench(name: str, buildScene: Callable[[], ScatteringScene], N: int) -> None:
    print(f"\n=== {name}  (N={N}) ===")
    # One warmup run per path so kernel build / IPP estimate don't pollute the timing.
    _runOnce(buildScene, N=max(500, N // 20), useBVH=False)
    _runOnce(buildScene, N=max(500, N // 20), useBVH=True)

    tFlat, tBuildFlat, nSolids, nTriangles, _ = _runOnce(buildScene, N=N, useBVH=False)
    tBVH, tBuildBVH, _, _, nNodes = _runOnce(buildScene, N=N, useBVH=True)

    tKernelFlat = max(tFlat - tBuildFlat, 1e-9)
    tKernelBVH = max(tBVH - tBuildBVH, 1e-9)
    propSpeedup = tKernelFlat / tKernelBVH
    totalSpeedup = tFlat / tBVH if tBVH > 0 else float("inf")

    print(f"  scene: {nSolids} solids, {nTriangles} triangles, BVH nodes: {nNodes}")
    print(f"  flat: build {tBuildFlat:6.3f}s + propagate {tKernelFlat:6.3f}s = total {tFlat:6.3f}s")
    print(f"  BVH:  build {tBuildBVH:6.3f}s + propagate {tKernelBVH:6.3f}s = total {tBVH:6.3f}s")
    print(f"  speedup — propagation only: {propSpeedup:5.2f}x    total (incl. build): {totalSpeedup:5.2f}x")


def exampleCode():
    if not hardwareAccelerationIsAvailable():
        print("Hardware acceleration is not available; the BVH benchmark requires OpenCL.")
        return

    _bench("Spherical shells (4 solids, ~970 tris)", _phantomShells, N=20_000)
    _bench("Cube with sphere inclusion (2 solids, ~1300 tris)", _cubeWithSphere, N=20_000)
    _bench("Two dense spheres (2 solids, ~2560 tris)", _denseSpheres, N=20_000)


TITLE = "BVH speedup vs flat per-solid AABB"
DESCRIPTION = (
    "Compares propagation time with the BVH disabled and enabled, splitting total runtime into "
    "BVH/scene construction vs. propagation kernel cost."
)


if __name__ == "__main__":
    exampleCode()
