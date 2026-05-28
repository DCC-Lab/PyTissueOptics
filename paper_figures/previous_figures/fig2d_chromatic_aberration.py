"""
Figure 2d -- Axial chromatic aberration via wavelength-dependent refractive index.

PyTissueOptics does not natively track wavelength on photons. For a steady-state
non-scattering experiment, a single wavelength is represented by running the
simulation with a fixed glass refractive index n(lambda). Re-running at three
Sellmeier-evaluated indices for Schott N-BK7 at 450, 550, 650 nm reveals axial
colour: the blue beam focuses at a shorter distance than the red beam, with a
relative focal shift approximately given by the thin-lens prediction
Delta_f / f ~= -Delta_n / (n - 1).

Setup:
  - SymmetricLens f=100 mm, d=25.4 mm, t=3.6 mm (default 2352-tri smoothed mesh)
  - Glass n at {1.5245 (450 nm), 1.5187 (550 nm), 1.5141 (650 nm)} from Sellmeier
  - Collimated beam, diameter 6 mm (narrow aperture to minimise SA and isolate chromatic)
  - 10^6 photons per wavelength
  - 13 axial screens spanning z in [95, 104] mm

Outputs: R90(z) per wavelength with best-focus marked, and a summary comparing
measured focal shift to the thin-lens Cauchy estimate.

Verification: (i) z_red > z_green > z_blue (normal dispersion for positive lens),
(ii) total axial shift z_red - z_blue within a factor of 2 of the thin-lens
Delta_f / f = -Delta_n / (n - 1) estimate.

Run:
    MPLBACKEND=Agg python fig2d_chromatic_aberration.py
    FIG2D_N=200000 python fig2d_chromatic_aberration.py    # quick check
"""
import os
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics import (
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
from pytissueoptics.scene.solids import ThickLens
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

N_PHOTONS_DEFAULT = 1_000_000 if hardwareAccelerationIsAvailable() else 5_000
N_PHOTONS = int(os.environ.get("FIG2D_N", N_PHOTONS_DEFAULT))
SEED = int(os.environ.get("FIG2D_SEED", "42"))

LENS_F = 100.0
LENS_D = 25.4
LENS_T = 3.6
BEAM_D = 6.0
SCREEN_Z = np.array([97.5, 98.25, 98.5, 98.7, 98.85, 99.0, 99.3, 99.7,
                     99.95, 100.1, 100.3, 100.55, 100.8, 101.0, 101.3])
SCREEN_SIDE = 30.0
SCREEN_THICK = 0.05


def bk7_sellmeier(lambda_um: float) -> float:
    """Schott N-BK7 Sellmeier coefficients (Schott datasheet), lambda in micrometres."""
    l2 = lambda_um ** 2
    n2 = (1.0
          + 1.03961212 * l2 / (l2 - 0.00600069867)
          + 0.231792344 * l2 / (l2 - 0.0200179144)
          + 1.01046945 * l2 / (l2 - 103.560653))
    return float(np.sqrt(n2))


WAVELENGTHS_NM = [450.0, 550.0, 650.0]
COLOURS = {450.0: "#2166ac", 550.0: "#4daf4a", 650.0: "#d62728"}
N_VALUES = {lam: bk7_sellmeier(lam / 1000.0) for lam in WAVELENGTHS_NM}

# Fix the lens geometry once, using the 550 nm index as reference, and then use
# this same (frontRadius, backRadius, diameter, thickness) for all wavelengths.
# Rationale: SymmetricLens(f, material) re-derives R from (f, n) so a larger n
# would give a flatter lens that still focuses at the same f, cancelling the
# chromatic shift we are trying to measure.
_REF_N = N_VALUES[550.0]
_p = np.sqrt(LENS_F * _REF_N * (LENS_F * _REF_N - LENS_T)) * (_REF_N - 1) / _REF_N
LENS_R = LENS_F * (_REF_N - 1) + _p


def run_for_index(n_value: float) -> Tuple[np.ndarray, List[str]]:
    glass = ScatteringMaterial(n=n_value)
    vacuum = ScatteringMaterial()

    # Fixed lens geometry: radii are derived once from the central-wavelength index
    # so that varying n genuinely changes the effective focal length. Using
    # SymmetricLens(f, n) at every lambda would re-solve the radii and cancel the
    # chromatic shift.
    lens = ThickLens(frontRadius=LENS_R, backRadius=-LENS_R, diameter=LENS_D,
                     thickness=LENS_T, material=glass, position=Vector(0, 0, 0))

    solids = [lens]
    screen_labels: List[str] = []
    for z in SCREEN_Z:
        label = f"Screen_z{z:.2f}"
        screen_labels.append(label)
        solids.append(Cuboid(a=SCREEN_SIDE, b=SCREEN_SIDE, c=SCREEN_THICK,
                             position=Vector(0, 0, z + SCREEN_THICK / 2),
                             material=vacuum, label=label))

    scene = ScatteringScene(solids)
    source = DirectionalSource(position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
                                diameter=BEAM_D, N=N_PHOTONS, displaySize=5, seed=SEED)
    logger = EnergyLogger(scene)
    source.propagate(scene, logger, showProgress=False)

    r90 = np.full(len(SCREEN_Z), np.nan)
    factory = PointCloudFactory(logger)
    for iz, label in enumerate(screen_labels):
        pc = factory.getPointCloud(label, f"{label}_front")
        pts = pc.enteringSurfacePoints
        if pts is None or len(pts) == 0:
            continue
        weights = np.abs(pts[:, 0])
        radii = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2)
        total = weights.sum()
        if total <= 0:
            continue
        order = np.argsort(radii)
        cum = np.cumsum(weights[order]) / total
        idx90 = int(np.searchsorted(cum, 0.90))
        idx90 = min(idx90, len(radii) - 1)
        r90[iz] = float(radii[order][idx90])
    return r90, screen_labels


def refine_best_focus(z_grid: np.ndarray, r90: np.ndarray) -> float:
    """Parabolic refinement around the argmin of R90(z) for sub-screen accuracy."""
    imin = int(np.nanargmin(r90))
    if imin == 0 or imin == len(z_grid) - 1:
        return float(z_grid[imin])
    z_lo, z_mid, z_hi = z_grid[imin - 1], z_grid[imin], z_grid[imin + 1]
    y_lo, y_mid, y_hi = r90[imin - 1], r90[imin], r90[imin + 1]
    denom = (y_lo - 2 * y_mid + y_hi)
    if denom == 0 or np.isnan(denom):
        return float(z_mid)
    shift = 0.5 * (y_lo - y_hi) / denom
    shift = max(min(shift, 1.0), -1.0)
    step = 0.5 * ((z_hi - z_lo))
    return float(z_mid + shift * step)


def main() -> int:
    print(f"[fig2d] N = {N_PHOTONS}, seed = {SEED}")
    for lam in WAVELENGTHS_NM:
        print(f"  lambda = {lam:.0f} nm  ->  n = {N_VALUES[lam]:.5f}")

    r90_per_lambda = {}
    for lam in WAVELENGTHS_NM:
        print(f"[fig2d] running lambda = {lam:.0f} nm (n = {N_VALUES[lam]:.5f})...", flush=True)
        r90, _ = run_for_index(N_VALUES[lam])
        r90_per_lambda[lam] = r90

    z_best = {lam: refine_best_focus(SCREEN_Z, r90_per_lambda[lam]) for lam in WAVELENGTHS_NM}

    z_green = z_best[550.0]
    n_center = N_VALUES[550.0]
    # Chromatic focal shift from thin-lens dispersion: Delta_f / f = -Delta_n / (n - 1).
    # Reference z_green (observed paraxial focus at the central wavelength) rather
    # than LENS_F, since BFL ~ 98.75 mm differs from the nominal f = 100 mm for a
    # thick lens. This isolates the chromatic shift from the thick-lens BFL offset.
    thin_lens_rel_shift = {
        lam: -z_green * (N_VALUES[lam] - n_center) / (n_center - 1.0)
        for lam in WAVELENGTHS_NM
    }

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)
    print(f"  {'lambda (nm)':<13} {'n':<10} {'z_best (mm)':<13} "
          f"{'Δz_meas (mm)':<15} {'Δz_thin-lens (mm)':<18}")
    print("  " + "-" * 70)
    for lam in WAVELENGTHS_NM:
        zm = z_best[lam]
        dz_meas = zm - z_green
        dz_pred = thin_lens_rel_shift[lam]
        print(f"  {lam:<13.0f} {N_VALUES[lam]:<10.5f} {zm:<13.3f} "
              f"{dz_meas:<+15.3f} {dz_pred:<+18.3f}")

    z_blue = z_best[450.0]
    z_red = z_best[650.0]
    measured_br_shift = z_red - z_blue
    predicted_br_shift = thin_lens_rel_shift[650.0] - thin_lens_rel_shift[450.0]

    print(f"\n  measured  z_red - z_blue = {measured_br_shift:+.3f} mm")
    print(f"  thin-lens z_red - z_blue = {predicted_br_shift:+.3f} mm "
          f"(using observed z_green = {z_green:.3f} mm as f)")

    results = []

    t1 = z_red > z_green > z_blue
    results.append(("normal dispersion: z_red > z_green > z_blue", t1,
                    f"z = ({z_blue:.3f}, {z_green:.3f}, {z_red:.3f}) mm"))

    if predicted_br_shift != 0:
        ratio = measured_br_shift / predicted_br_shift
    else:
        ratio = float("nan")
    t2 = 0.5 < ratio < 2.0
    results.append(("total chromatic shift within factor 2 of thin-lens estimate", t2,
                    f"ratio = {ratio:.2f}"))

    t3 = measured_br_shift > 0
    results.append(("positive axial chromatic shift (blue focuses shorter)", t3,
                    f"shift = {measured_br_shift:+.3f} mm"))

    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  -- {detail}")

    all_pass = all(ok for _, ok, _ in results)
    print("\n" + "=" * 72)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    print("=" * 72)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for lam in WAVELENGTHS_NM:
        ax.plot(SCREEN_Z, r90_per_lambda[lam], "o-", ms=4, lw=1.4,
                color=COLOURS[lam],
                label=f"{lam:.0f} nm, n={N_VALUES[lam]:.4f}")
        ax.axvline(z_best[lam], color=COLOURS[lam], ls="--", lw=1.0, alpha=0.8)
    ax.axvline(LENS_F, color="k", ls=":", lw=0.6, alpha=0.4, label=f"nominal f = {LENS_F:.0f} mm")
    ax.axvline(z_green, color="0.3", ls=":", lw=0.8, alpha=0.7,
               label=f"observed z(green) = {z_green:.2f} mm")
    ax.set_xlabel("axial position z (mm)")
    ax.set_ylabel("R90 spot radius (mm)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")

    out_dir = os.environ.get("FIG2D_OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "fig2d_chromatic_aberration.pdf")
    out_png = os.path.join(out_dir, "fig2d_chromatic_aberration.png")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n[save] {out_pdf}")
    print(f"[save] {out_png}")

    if os.environ.get("FIG2D_SHOW_3D"):
        _show_3d_visualization()

    return 0 if all_pass else 1


def _show_3d_visualization():
    """Interactive 3D view for the central wavelength. Needs a Mayavi display."""
    N_3D = int(os.environ.get("FIG2D_N_3D", "20000"))
    lam = float(os.environ.get("FIG2D_3D_LAMBDA", "550"))
    n_value = N_VALUES.get(lam, bk7_sellmeier(lam / 1000.0))

    glass = ScatteringMaterial(n=n_value)
    vacuum = ScatteringMaterial()

    lens = ThickLens(frontRadius=LENS_R, backRadius=-LENS_R, diameter=LENS_D,
                     thickness=LENS_T, material=glass, position=Vector(0, 0, 0))
    screens = []
    for z in [95.0, 98.0, 99.0, 100.0, 101.0, 105.0]:
        screens.append(Cuboid(a=SCREEN_SIDE, b=SCREEN_SIDE, c=SCREEN_THICK,
                              position=Vector(0, 0, z + SCREEN_THICK / 2),
                              material=vacuum, label=f"screen_{z:.1f}"))

    scene = ScatteringScene([lens, *screens])
    source = DirectionalSource(position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
                                diameter=BEAM_D, N=N_3D, displaySize=5, seed=SEED)

    print(f"\n[3D] Launching {N_3D} photons at lambda = {lam} nm (n = {n_value:.5f}) ...")
    scene.show(source=source)
    logger = EnergyLogger(scene)
    source.propagate(scene, logger)

    viewer = Viewer(scene, source, logger)
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))


if __name__ == "__main__":
    sys.exit(main())
