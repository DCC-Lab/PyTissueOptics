"""Publication figure: through-focus R90 per entrance-pupil zone (spherical aberration).

Single panel, paired with fig_paper_chromatic.py.  Runs one MC simulation on a
SymmetricLens f=100 mm, n=1.50, d=25.4 mm, t=3.6 mm with a d=20 mm collimated
beam and plots R90(z) for five entrance-pupil rings ρ_0 ∈ [0,2), ..., [8,10) mm.
Each ring's best-focus (argmin of R90) sits at a different z — that shift IS
the longitudinal spherical aberration of a spherical biconvex lens.

Physics behind the plot
-----------------------
A spherical refracting surface cannot focus a parallel bundle to a single point:
rays that enter at large height rho_0 "over-bend" and cross the axis at a
smaller z than paraxial rays.  This is *primary* spherical aberration (Seidel,
third order).  If we label each photon by its entrance-pupil radius rho_0 and
look at the spot size as a function of axial position z for each annular zone
of rho_0, we see five V-shaped curves whose minima slide to smaller z with
larger rho_0.  The magnitude of that slide (mm shift per mm^2 of rho_0^2) is
the primary-SA coefficient k_2.

Environment variables
---------------------
N       : number of photons (default 300k on GPU, 5k on CPU).
NO_3D=1 : skip interactive 3D viewers at the end.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pytissueoptics import (
    Cuboid, DirectionalSource, EnergyLogger, ScatteringMaterial,
    ScatteringScene, SymmetricLens, Vector, hardwareAccelerationIsAvailable,
)
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# Photon count: use 300k with GPU (propagates in <1 s on an M-series), drop to
# 5k on CPU just so the script runs without hardware acceleration.  Fixed seed
# keeps ring assignments reproducible between runs.
N = int(os.environ.get("N", 300_000 if hardwareAccelerationIsAvailable() else 5_000))
SEED = 42

# Lens and beam geometry.
#   LENS_F : paraxial focal length (mm).
#   LENS_D : lens clear aperture diameter (mm) — 25.4 mm = 1 inch (standard).
#   LENS_T : centre thickness (mm); required because SymmetricLens is *thick*.
#   GLASS_N: refractive index.  1.50 is a generic crown-glass value.
#   BEAM_D : collimated beam diameter (mm).  20 mm < 25.4 mm so the beam fills
#            most of the aperture without over-filling (avoids edge losses).
LENS_F, LENS_D, LENS_T, GLASS_N, BEAM_D = 100.0, 25.4, 3.6, 1.50, 20.0

# Screen grid: dense through the focal caustic (every 0.1-0.15 mm from z=98.75
# to z=100.7 mm) so ring minima are resolved; sparse outside to keep the total
# screen count low.  Each screen logs photon crossings, so more screens = more
# memory + slower kernel.
Z_SCREENS = np.array([96.0, 97.0, 98.0, 98.5, 98.75, 99.0, 99.1, 99.25, 99.4,
                      99.55, 99.7, 99.8, 99.9, 100.0, 100.1, 100.2, 100.3,
                      100.5, 100.7, 101.0, 101.5, 102.5])

# Entrance-pupil rings: concentric annuli of entrance height.  Five rings of
# 2-mm width uniformly partition the beam radius [0, 10) mm.  The INNER ring
# [0, 2) is statistics-limited because uniform-disc sampling puts only ~4% of
# photons there, so we mostly rely on the outer four for the SA fit.
RINGS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]

# ---------------------------------------------------------------------------
# 2. Build the scene
# ---------------------------------------------------------------------------
# Materials.
#   - `glass`  : refractive, non-scattering, non-absorbing (mu_s=mu_a=0 default).
#   - `vacuum` : n=1, no interaction.  Used for the screens so they don't
#                refract passing photons.
glass, vacuum = ScatteringMaterial(n=GLASS_N), ScatteringMaterial()

# The SymmetricLens constructor SOLVES the lensmaker equation for you: given
# (f, diameter, thickness, n), it picks the radius of curvature R that makes
# the lens focus at f.  Default primitive = 2352 triangles with normal
# smoothing, adequate for f-number ~5 without visible faceting.
lens = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                     material=glass, position=Vector(0, 0, 0))

# Screens are thin vacuum cuboids perpendicular to z.  Each screen has an
# identifying label used later to retrieve its logged crossings.  Cuboid
# thickness = 0.05 mm; we centre the cuboid at z + 0.025 so its FRONT face
# lies exactly at z (simplifies the "R90 at z" interpretation).
screens, labels = [], []
for z in Z_SCREENS:
    lbl = f"s{z:.2f}"
    screens.append(Cuboid(40, 40, 0.05, position=Vector(0, 0, z + 0.025),
                          material=vacuum, label=lbl))
    labels.append(lbl)

# Combine lens + screens into the scene.  The 40x40 mm screen footprint is
# much larger than the ~2 mm beam near focus, so photons never miss a screen
# laterally.  `ScatteringScene` will error if any solids overlap — our screens
# are 0.1 mm apart >> 0.05 mm thickness so no collision.
scene = ScatteringScene([lens, *screens])

# Collimated 20-mm beam starting at z=-20 mm (well upstream of the lens at z=0).
# `seed` fixes the per-photon launch positions so r0 is deterministic.
src = DirectionalSource(Vector(0, 0, -20), Vector(0, 0, 1), diameter=BEAM_D,
                        N=N, displaySize=5, seed=SEED)

# Entrance pupil radius per photon: rho_0 = sqrt(x_launch^2 + y_launch^2).
# The source stores launch positions internally; we read them BEFORE
# propagation because the OpenCL kernel may re-use that buffer.  Each photon
# retains its ID through propagation, so we can later look up its rho_0 by ID.
r0 = np.hypot(src._photons._positions[:, 0], src._photons._positions[:, 1])

# ---------------------------------------------------------------------------
# 3. Propagate
# ---------------------------------------------------------------------------
# EnergyLogger(scene) with default settings keeps 3D data (one row per
# interaction: [value, x, y, z, photon_ID]).  `showProgress=False` silences
# the GPU progress bar so the script output is clean.
logger = EnergyLogger(scene)
src.propagate(scene, logger, showProgress=False)

# ---------------------------------------------------------------------------
# 4. Compute R90 per ring per screen
# ---------------------------------------------------------------------------
# R90 = radius enclosing 90% of the energy-weighted photon hits at a given
# screen.  Computed for each (ring, screen) combination.  A ring's best-focus
# is the argmin over screens of R90, and the shift of that argmin with ring
# radius is the spherical aberration.
factory = PointCloudFactory(logger)
r90 = np.full((len(RINGS), len(Z_SCREENS)), np.nan)

for iz, lbl in enumerate(labels):
    # `enteringSurfacePoints` gives all photon crossings INTO this screen's
    # front face, as rows [value, x, y, z, photon_ID].  `value` is negative
    # (sign convention: <0 = entering), so we take abs() later.
    pts = factory.getPointCloud(lbl, f"{lbl}_front").enteringSurfacePoints
    if pts is None or pts.shape[1] < 5:
        continue

    # Map screen-crossing to its photon's entrance radius via photon ID.
    ids = pts[:, 4].astype(np.int64)
    for ir, (lo, hi) in enumerate(RINGS):
        # Mask: only hits whose photon_ID has rho_0 in [lo, hi).
        mask = (r0[ids] >= lo) & (r0[ids] < hi)
        if mask.sum() < 10:
            continue   # too few photons for a meaningful R90

        # Radial distance from optical axis at this screen, for ring photons.
        r = np.hypot(pts[mask, 1], pts[mask, 2])
        # Photon weight at crossing (positive, magnitude of the signed `value`).
        w = np.abs(pts[mask, 0])

        # R90: sort by radius, find the radius where cumulative weight = 90%.
        order = np.argsort(r)
        cum = np.cumsum(w[order])
        idx = int(np.searchsorted(cum, 0.9 * cum[-1]))
        r90[ir, iz] = r[order][min(idx, len(r) - 1)]

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
# Aspect ratio ~3:2 is a good single-column figure size for a paper.
fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
cmap = plt.get_cmap("viridis")
markers = ["o", "s", "^", "D", "v"]

# Print per-ring best-focus for verification / caption text.
for ir, (lo, hi) in enumerate(RINGS):
    if np.isfinite(r90[ir]).any():
        imin = int(np.nanargmin(r90[ir]))
        print(f"  ring ρ ∈ [{lo},{hi}) mm : "
              f"min R90 = {r90[ir, imin]*1e3:.1f} µm at z = {Z_SCREENS[imin]:.2f} mm")

# Draw OUTER rings first (larger R90) so INNER rings (smaller R90, more
# informative minima) render on top — otherwise the yellow outer curve covers
# the violet inner curve where they overlap.
for ir, (lo, hi) in reversed(list(enumerate(RINGS))):
    ax.plot(Z_SCREENS, r90[ir], "-" + markers[ir], ms=4.0, lw=1.4,
            color=cmap(ir / (len(RINGS) - 1)),
            label=rf"$\rho_0 \in [{lo},\,{hi})$ mm",
            zorder=10 - ir)

# Nominal paraxial focal length reference line.
ax.axvline(LENS_F, color="k", ls=":", lw=0.8, alpha=0.5, zorder=0)

# Log y-scale: R90 spans 1 µm to 100 µm across the data, so log is clearer
# than linear.
ax.set_xlabel(r"axial position $z$ (mm)", fontsize=10)
ax.set_ylabel(r"$R_{90}$ spot radius (mm)", fontsize=10)
ax.set_yscale("log")
ax.set_xlim(96, 103)
ax.set_ylim(1e-3, 0.1)
ax.tick_params(axis="both", labelsize=9)
ax.grid(alpha=0.3, which="both")
# Legend OUTSIDE the plot (to the right) so curves never touch it.
ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False, handletextpad=0.4, borderpad=0.3)

# Save as both PDF (for the paper) and PNG (for quick preview).
out = os.path.dirname(os.path.abspath(__file__))
for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = os.path.join(out, f"fig_paper_sa.{ext}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save] {path}")

# ---------------------------------------------------------------------------
# 6. Optional 3D visualization (Mayavi, interactive — needs a display)
# ---------------------------------------------------------------------------
# Skipped when NO_3D=1 or MPLBACKEND=Agg pipelines.  Running locally just
# `python fig_paper_sa.py` will (i) pop the scene+source geometry, then (ii)
# pop the post-propagation photon point cloud.
if os.environ.get("NO_3D") != "1":
    from pytissueoptics import Viewer, PointCloudStyle
    print("\n[3D] Showing setup (close window to continue) ...")
    scene.show(source=src)
    print("[3D] Showing post-propagation photon cloud (close window to exit) ...")
    Viewer(scene, src, logger).show3D(
        pointCloudStyle=PointCloudStyle(showSolidPoints=False))
