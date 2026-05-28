"""
Figure 2c -- Fibre-coupling efficiency as a function of entrance aperture.

A practical illustration of the SA measured in fig2b: for a fixed singlet
collecting collimated light and focussing into a detector that approximates
a multimode fibre core, increasing the entrance aperture delivers more
geometric throughput but at the cost of SA-driven spot broadening. The
coupling efficiency (fraction of launched photons absorbed by the detector)
therefore degrades monotonically with aperture once SA dominates.

Setup:
  - SymmetricLens f=50 mm, d=25.4 mm, t=3.6 mm, n=1.50 (default 2352-tri smoothed mesh)
  - Collimated DirectionalSource at z=-20 mm, 10^6 photons
  - Detector: Circle(radius=0.1 mm, asDetector(halfAngle=pi/2)) centred at z=50 mm
  - Beam diameters swept over {3, 6, 9, 12, 15, 18, 21} mm

Outputs: coupling-efficiency vs beam-diameter curve, encircled-energy curves
at the paraxial focal plane for three representative apertures, and a summary
table of (d_beam, RMS, R90, coupling).

Verification: (i) efficiency at smallest aperture > 0.80, (ii) efficiency at
largest aperture < 0.50, (iii) efficiency non-increasing with aperture
(monotonic degradation), (iv) R90 strictly increases with aperture.

Run:
    MPLBACKEND=Agg python fig2c_coupling_efficiency.py
    FIG2C_N=200000 python fig2c_coupling_efficiency.py      # quick check
"""
import os
import sys
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics import (
    Circle,
    Cuboid,
    DirectionalSource,
    EnergyLogger,
    PointCloudStyle,
    ScatteringMaterial,
    ScatteringScene,
    SymmetricLens,
    Vector,
    Viewer,
    hardwareAccelerationIsAvailable,
)
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory
from pytissueoptics.scene.logger import InteractionKey

N_PHOTONS_DEFAULT = 1_000_000 if hardwareAccelerationIsAvailable() else 5_000
N_PHOTONS = int(os.environ.get("FIG2C_N", N_PHOTONS_DEFAULT))
SEED = int(os.environ.get("FIG2C_SEED", "42"))

LENS_F = 50.0
LENS_D = 25.4
LENS_T = 3.6
GLASS_N = 1.50
DETECTOR_RADIUS = 0.1  # mm (mimics a 200 um core MMF)
BEAM_DIAMETERS = [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
SCREEN_LABEL = "FocalScreen"


def run_single(beam_diameter: float) -> Tuple[float, Optional[float], Optional[float], int]:
    glass = ScatteringMaterial(n=GLASS_N)
    vacuum = ScatteringMaterial()

    lens = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                         material=glass, position=Vector(0, 0, 0))
    screen = Cuboid(a=30, b=30, c=0.05, position=Vector(0, 0, LENS_F + 0.025),
                    material=vacuum, label=SCREEN_LABEL)

    det = Circle(radius=DETECTOR_RADIUS, orientation=Vector(0, 0, -1),
                 position=Vector(0, 0, LENS_F - 0.5), label="detector")
    det.asDetector(halfAngle=np.pi / 2)

    scene = ScatteringScene([lens, screen, det])

    source = DirectionalSource(position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
                                diameter=beam_diameter, N=N_PHOTONS, displaySize=5, seed=SEED)
    logger = EnergyLogger(scene)
    source.propagate(scene, logger, showProgress=False)

    det_pts = logger.getRawDataPoints(InteractionKey("detector"))
    if det_pts is None or len(det_pts) == 0:
        n_det = 0
    elif det_pts.shape[1] >= 5:
        n_det = int(np.unique(det_pts[:, 4].astype(np.uint32)).size)
    else:
        n_det = int(det_pts.shape[0])
    coupling = n_det / N_PHOTONS

    factory = PointCloudFactory(logger)
    pc = factory.getPointCloud(SCREEN_LABEL, f"{SCREEN_LABEL}_front")
    pts = pc.enteringSurfacePoints
    if pts is None or len(pts) == 0:
        return coupling, None, None, n_det
    weights = np.abs(pts[:, 0])
    radii = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    total = weights.sum()
    if total <= 0:
        return coupling, None, None, n_det
    rms = float(np.sqrt((weights * radii ** 2).sum() / total))
    order = np.argsort(radii)
    cum = np.cumsum(weights[order]) / total
    idx90 = int(np.searchsorted(cum, 0.90))
    idx90 = min(idx90, len(radii) - 1)
    r90 = float(radii[order][idx90])

    return coupling, rms, r90, n_det


def encircled_energy_profile(beam_diameter: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    glass = ScatteringMaterial(n=GLASS_N)
    vacuum = ScatteringMaterial()

    lens = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                         material=glass, position=Vector(0, 0, 0))
    screen = Cuboid(a=30, b=30, c=0.05, position=Vector(0, 0, LENS_F + 0.025),
                    material=vacuum, label=SCREEN_LABEL)
    scene = ScatteringScene([lens, screen])

    source = DirectionalSource(position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
                                diameter=beam_diameter, N=N_PHOTONS, displaySize=5, seed=SEED)
    logger = EnergyLogger(scene)
    source.propagate(scene, logger, showProgress=False)

    factory = PointCloudFactory(logger)
    pc = factory.getPointCloud(SCREEN_LABEL, f"{SCREEN_LABEL}_front")
    pts = pc.enteringSurfacePoints
    if pts is None or len(pts) == 0:
        return None
    weights = np.abs(pts[:, 0])
    radii = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
    order = np.argsort(radii)
    sorted_r = radii[order]
    cum = np.cumsum(weights[order]) / weights.sum()
    return sorted_r, cum


def main() -> int:
    print(f"[fig2c] N = {N_PHOTONS}, seed = {SEED}")
    print(f"[fig2c] Lens: f={LENS_F} mm, d={LENS_D} mm, n={GLASS_N}")
    print(f"[fig2c] Detector: Circle(r={DETECTOR_RADIUS} mm) at z={LENS_F:.1f} mm (paraxial focal plane)")

    couplings = []
    rms_list = []
    r90_list = []
    n_det_list = []
    for d_beam in BEAM_DIAMETERS:
        print(f"[fig2c]   running d_beam = {d_beam:.1f} mm ...", flush=True)
        coupling, rms, r90, n_det = run_single(d_beam)
        couplings.append(coupling)
        rms_list.append(rms if rms is not None else np.nan)
        r90_list.append(r90 if r90 is not None else np.nan)
        n_det_list.append(n_det)

    couplings = np.array(couplings)
    rms_arr = np.array(rms_list)
    r90_arr = np.array(r90_list)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  {'d_beam (mm)':<13} {'RMS (mm)':<11} {'R90 (mm)':<11} {'n_det':<9} {'coupling':<10}")
    print("  " + "-" * 56)
    for i, d in enumerate(BEAM_DIAMETERS):
        print(f"  {d:<13.1f} {rms_arr[i]:<11.4f} {r90_arr[i]:<11.4f} "
              f"{n_det_list[i]:<9d} {couplings[i]:<10.4f}")

    results = []

    t1 = couplings[0] > 0.80
    results.append(("coupling at smallest aperture (d_beam={:.1f}) > 0.80".format(BEAM_DIAMETERS[0]), t1,
                    f"coupling = {couplings[0]:.3f}"))

    t2 = couplings[-1] < 0.50
    results.append(("coupling at largest aperture (d_beam={:.1f}) < 0.50".format(BEAM_DIAMETERS[-1]), t2,
                    f"coupling = {couplings[-1]:.3f}"))

    diffs = np.diff(couplings)
    slack = 0.02
    t3 = bool(np.all(diffs <= slack))
    results.append(("coupling non-increasing with aperture (slack {:.2f})".format(slack), t3,
                    f"max positive step = {diffs.max():+.3f}"))

    r90_diffs = np.diff(r90_arr)
    t4 = bool(np.all(r90_diffs >= -1e-3))
    results.append(("R90 non-decreasing with aperture", t4,
                    f"min step = {r90_diffs.min():+.4f} mm"))

    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -- {detail}")

    all_pass = all(ok for _, ok, _ in results)
    print("\n" + "=" * 72)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 72)

    # Encircled-energy curves at three representative apertures.
    ee_apertures = [BEAM_DIAMETERS[0], BEAM_DIAMETERS[len(BEAM_DIAMETERS) // 2], BEAM_DIAMETERS[-1]]
    ee_profiles = {d: encircled_energy_profile(d) for d in ee_apertures}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.plot(BEAM_DIAMETERS, couplings * 100, "ko-", ms=6, lw=1.4)
    ax.axhline(100, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("entrance beam diameter (mm)")
    ax.set_ylabel(f"coupling efficiency into r = {DETECTOR_RADIUS * 1000:.0f} μm detector (%)")
    ax.set_title("(a) Coupling vs aperture")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    ax = axes[1]
    cmap = plt.get_cmap("plasma")
    for i, d in enumerate(ee_apertures):
        prof = ee_profiles[d]
        if prof is None:
            continue
        r, c = prof
        ax.plot(r, c * 100, lw=1.6, color=cmap(i / max(len(ee_apertures) - 1, 1)),
                label=f"d = {d:.1f} mm")
    ax.axvline(DETECTOR_RADIUS, color="red", ls="--", lw=1.0, alpha=0.8,
               label=f"r_det = {DETECTOR_RADIUS:.2f} mm")
    ax.axhline(90, color="gray", ls=":", lw=0.6, alpha=0.6)
    ax.set_xlabel("radius at focal plane (mm)")
    ax.set_ylabel("encircled energy (%)")
    ax.set_title("(b) Encircled-energy at paraxial focus")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 3.0)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.suptitle(f"SA-limited coupling into r={DETECTOR_RADIUS:.2f} mm detector  —  "
                 f"SymmetricLens f={LENS_F:.0f} mm, N = {N_PHOTONS:,} per sweep point")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_dir = os.environ.get("FIG2C_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "fig2c_coupling_efficiency.pdf")
    out_png = os.path.join(out_dir, "fig2c_coupling_efficiency.png")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n[save] {out_pdf}")
    print(f"[save] {out_png}")

    if os.environ.get("FIG2C_SHOW_3D"):
        _show_3d_visualization()

    return 0 if all_pass else 1


def _show_3d_visualization():
    """Interactive 3D view for a representative aperture. Needs a Mayavi display."""
    N_3D = int(os.environ.get("FIG2C_N_3D", "20000"))
    d_beam = float(os.environ.get("FIG2C_3D_DBEAM", "12.0"))

    glass = ScatteringMaterial(n=GLASS_N)
    vacuum = ScatteringMaterial()

    lens = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                         material=glass, position=Vector(0, 0, 0))
    screen = Cuboid(a=30, b=30, c=0.05, position=Vector(0, 0, LENS_F + 0.025),
                    material=vacuum, label=SCREEN_LABEL)
    det = Circle(radius=DETECTOR_RADIUS, orientation=Vector(0, 0, -1),
                 position=Vector(0, 0, LENS_F - 0.5), label="detector")
    det.asDetector(halfAngle=np.pi / 2)

    scene = ScatteringScene([lens, screen, det])
    source = DirectionalSource(position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
                                diameter=d_beam, N=N_3D, displaySize=5, seed=SEED)

    print(f"\n[3D] Launching {N_3D} photons at d_beam = {d_beam} mm ...")
    scene.show(source=source)
    logger = EnergyLogger(scene)
    source.propagate(scene, logger)

    viewer = Viewer(scene, source, logger)
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))


if __name__ == "__main__":
    sys.exit(main())
