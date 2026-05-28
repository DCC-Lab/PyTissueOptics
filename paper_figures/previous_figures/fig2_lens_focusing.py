"""
Figure 2 – Symmetric thick lens focusing a collimated beam.

(a) 3D point cloud of photon trajectories showing convergence toward the focal region.
(b) Encircled-energy profile at the focal plane: smoothed vs unsmoothed mesh,
    multiple resolutions.

Verifies that normal smoothing on tessellated curved surfaces correctly reproduces
geometric-optics focusing behavior.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics import *  # noqa: F403
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

TITLE = "Symmetric thick lens focusing a collimated beam (non-scattering)"

DESCRIPTION = """ A collimated beam of 10^6 photons is focused by a SymmetricLens (f=100 mm, n=1.50)
with normal-smoothed tessellated surfaces. Screens at various distances capture the transverse
intensity profile. The focal spot size is verified against the geometric optics expectation. """


def count_polygons(lens):
    """Count total polygons on the lens mesh."""
    total = 0
    for surface in ["front", "back", "lateral"]:
        total += len(lens.getPolygons(surface))
    return total


def get_encircled_energy(logger, screen_label):
    """Extract radial encircled-energy curve from a screen surface."""
    factory = PointCloudFactory(logger)
    surface_label = f"{screen_label}_front"
    pc = factory.getPointCloud(screen_label, surface_label)
    entering = pc.enteringSurfacePoints
    if entering is None or len(entering) == 0:
        return None, None, None, None

    energies = np.abs(entering[:, 0])
    x_pos = entering[:, 1]
    y_pos = entering[:, 2]
    radii = np.sqrt(x_pos**2 + y_pos**2)

    total_energy = np.sum(energies)
    rms_radius = np.sqrt(np.sum(energies * radii**2) / total_energy)

    sorted_idx = np.argsort(radii)
    sorted_radii = radii[sorted_idx]
    cumulative = np.cumsum(energies[sorted_idx]) / total_energy

    r90_idx = np.searchsorted(cumulative, 0.90)
    r90 = sorted_radii[r90_idx] if r90_idx < len(sorted_radii) else sorted_radii[-1]

    return sorted_radii, cumulative, rms_radius, r90


def run_single_lens(f, beam_diameter, glass_n, N, u=24, v=2, s=24, smooth=True):
    """Run a single lens simulation and return logger + polygon count."""
    glassMaterial = ScatteringMaterial(n=glass_n)
    vacuum = ScatteringMaterial()

    lens = SymmetricLens(f=f, diameter=25.4, thickness=3.6, material=glassMaterial,
                         position=Vector(0, 0, 0), u=u, v=v, s=s, smooth=smooth)
    n_polygons = count_polygons(lens)

    screen_focal = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, f + 0.05),
                          material=vacuum, label="FocalScreen")

    scene = ScatteringScene([screen_focal, lens])
    logger = EnergyLogger(scene)

    source = DirectionalSource(
        position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
        diameter=beam_diameter, N=N, displaySize=5
    )
    source.propagate(scene, logger)

    return logger, n_polygons


def plot_encircled_energy(results, savepath="fig_lens_psf.pdf"):
    """Plot encircled-energy curves for multiple configurations."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for label, radii, cumulative, linestyle in results:
        ax.plot(radii, cumulative * 100, linestyle, label=label, linewidth=1.5)

    # 90% reference: dotted line spanning the axis; label sits at the right edge
    # (the curves have already saturated there, so nothing to overlap).
    ax.axhline(90, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.text(1.98, 91, '90%', color='gray', fontsize=9, ha='right', va='bottom')

    ax.set_xlabel("Radius (mm)")
    ax.set_ylabel("Encircled energy (%)")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc='lower right')
    fig.tight_layout()
    fig.savefig(savepath, dpi=300)
    print(f"  Saved: {savepath}")
    if not os.environ.get("LENS_PSF_ONLY"):
        plt.show()


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 2000

    f = 100
    beam_diameter = 20.0
    glass_n = 1.50

    # Run multiple configurations for comparison. Only u (azimuthal wedges
    # around the optical axis) sets the curved-surface tessellation that
    # affects the wavefront; v and s (longitudinal slabs and radial rings on
    # the side and caps) are mesh-density knobs that don't change the PSF.
    # Legend labels report the triangle count — the quantity that matters.
    configs = [
        ("576 triangles, smoothed",    12, 1, 12, True),
        ("2 352 triangles, smoothed",  24, 2, 24, True),
        ("9 504 triangles, smoothed",  48, 4, 48, True),
        ("2 352 triangles, flat",      24, 2, 24, False),
    ]

    ee_results = []
    print("\n" + "=" * 70)
    print("FOCAL SPOT COMPARISON")
    print("=" * 70)
    print(f"  {'Configuration':<30} {'Triangles':>10} {'RMS (mm)':>10} {'R90 (mm)':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")

    for label, u, v, s, smooth in configs:
        logger, n_poly = run_single_lens(f, beam_diameter, glass_n, N, u, v, s, smooth)
        radii, cumulative, rms, r90 = get_encircled_energy(logger, "FocalScreen")
        print(f"  {label:<30} {n_poly:>10} {rms:>10.4f} {r90:>10.4f}")

        linestyle = "--" if not smooth else "-"
        ee_results.append((label, radii, cumulative, linestyle))

    print("=" * 70 + "\n")

    # Plot encircled energy comparison
    plot_encircled_energy(ee_results)

    # Skip the full-scene 3D / 2D-screen visualization when we only want the
    # paper PSF figure. The PDF is already written by plot_encircled_energy.
    if os.environ.get("LENS_PSF_ONLY"):
        return

    # Run the full scene with screens for the 3D visualization and 2D views
    print("\nRunning full scene for visualization...")
    glassMaterial = ScatteringMaterial(n=glass_n)
    vacuum = ScatteringMaterial()

    lens = SymmetricLens(f=f, diameter=25.4, thickness=3.6, material=glassMaterial,
                         position=Vector(0, 0, 0))
    screen_pre = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, 30),
                        material=vacuum, label="Screen30mm")
    screen_mid = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, 70),
                        material=vacuum, label="Screen70mm")
    screen_focal = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, f + 0.05),
                          material=vacuum, label="FocalScreen")

    scene = ScatteringScene([screen_pre, screen_mid, screen_focal, lens])
    source = DirectionalSource(
        position=Vector(0, 0, -20), direction=Vector(0, 0, 1),
        diameter=beam_diameter, N=N, displaySize=5
    )

    scene.show(source)

    logger = EnergyLogger(scene)
    source.propagate(scene, logger)

    viewer = Viewer(scene, source, logger)
    viewer.reportStats()

    viewer.show3D()
    viewer.show2D(View2DSurfaceZ("Screen30mm", "Screen30mm_front", surfaceEnergyLeaving=False))
    viewer.show2D(View2DSurfaceZ("Screen70mm", "Screen70mm_front", surfaceEnergyLeaving=False))
    viewer.show2D(View2DSurfaceZ("FocalScreen", "FocalScreen_front", surfaceEnergyLeaving=False))


if __name__ == "__main__":
    exampleCode()
