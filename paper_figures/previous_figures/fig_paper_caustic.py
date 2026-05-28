"""Publication figure: meridional caustic of a biconvex thick lens.

Standalone counterpart to fig_paper_sa.py.  Same lens and beam, but instead of
plotting R90(z) per entrance-pupil zone we draw the continuous ray bundle in
the meridional (x, z) plane.  Rays at large entrance height rho_0 (yellow)
visibly cross the axis before rays near the axis (violet), producing the
classic "teacup" caustic that IS the signature of primary spherical aberration.

How the "continuous" rays are drawn
-----------------------------------
PyTissueOptics logs a photon's (x, y, z) only at the surfaces it crosses.  For
this non-scattering scene the chosen surfaces are thin vacuum screens at 19
axial positions.  Between any two screens the photon travels in a straight line
(no scattering, no absorption), so joining the screen crossings of a given
photon by straight line segments gives its *exact* trajectory — no
interpolation error.  Doing this for ~1200 photons drawn from a meridional
slab (|y_0| < 0.5 mm) produces the publication figure.

Why inverse-rho balancing
-------------------------
Uniform-disc illumination (DirectionalSource) puts ~36% of photons in the
outer ring [8, 10) mm and only ~4% in the paraxial ring [0, 2) mm.  To keep
inner and outer zones visually balanced we pick ~60 rays from each of 20 equal
rho_0 bins, regardless of the underlying PDF.  The final plot therefore shows
rays uniformly spread in rho_0, not in photon count.

Environment variables
---------------------
N       : number of photons launched (default 500k on GPU).
NO_3D=1 : skip interactive 3D viewers at the end.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from pytissueoptics import (
    Cuboid, DirectionalSource, EnergyLogger, ScatteringMaterial,
    ScatteringScene, SymmetricLens, Vector, hardwareAccelerationIsAvailable,
)
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# 500k photons is plenty for ~1200 selected meridional rays; the rest are just
# pool we sample from.  Fixed seed for reproducibility of which rays are drawn.
N = int(os.environ.get("N", 500_000 if hardwareAccelerationIsAvailable() else 5_000))
SEED = 42

# Lens + beam (match fig_paper_sa.py so the two figures describe the same
# experiment).
LENS_F, LENS_D, LENS_T, GLASS_N, BEAM_D = 100.0, 25.4, 3.6, 1.50, 20.0

# Meridional slab half-width: only photons launched with |y_0| < this value are
# kept.  Restricting to a thin slab lets us plot (z, x) as a 2D slice without
# projection artefacts — a photon with y_0 ~ 5 mm projected onto the x-z plane
# would mix its radial displacement into x and break the meridional picture.
MERIDIONAL_SLAB = 0.5   # mm

# 19 screens: sparse in the converging tails (every 2 mm from 85 to 95 mm),
# dense through the focal caustic (0.2-0.25 mm spacing from z=98 to z=101).
# More screens would give smoother trajectories but balloon kernel memory.
Z_SCREENS = np.array([85.0, 87.0, 89.0, 91.0, 93.0, 95.0, 96.5, 97.5,
                      98.25, 98.75, 99.1, 99.4, 99.7, 99.9, 100.1, 100.3,
                      100.6, 101.0, 102.0])

# ---------------------------------------------------------------------------
# 2. Build the scene
# ---------------------------------------------------------------------------
# `vacuum` has n=1, mu_s=mu_a=0 (nothing happens inside).  The screens are
# therefore invisible to a photon except as logging planes — when the photon
# crosses one, the logger writes a row [value, x, y, z, photon_ID].
glass, vacuum = ScatteringMaterial(n=GLASS_N), ScatteringMaterial()

# SymmetricLens solves the thick-lens equation for you: given (f, D, t, n) it
# picks the sphere radius R such that 1/f = (n-1)[2/R - (n-1)t/(nR^2)].
lens = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                     material=glass, position=Vector(0, 0, 0))

# Screens are thin (0.05 mm) vacuum cuboids.  The factor of 2 between this
# thickness and the minimum 0.1-mm screen spacing avoids the
# "partially-overlapping-solids" error that ScatteringScene raises.
screens, labels = [], []
for z in Z_SCREENS:
    lbl = f"s{z:.2f}"
    screens.append(Cuboid(40, 40, 0.05, position=Vector(0, 0, z + 0.025),
                          material=vacuum, label=lbl))
    labels.append(lbl)
scene = ScatteringScene([lens, *screens])

# Collimated 20 mm beam starting 20 mm upstream of the lens.  Seed fixes the
# per-photon launch (x, y) positions, which determine rho_0 = sqrt(x^2+y^2).
src = DirectionalSource(Vector(0, 0, -20), Vector(0, 0, 1), diameter=BEAM_D,
                        N=N, displaySize=5, seed=SEED)

# Entrance-pupil radius per photon (evaluated BEFORE propagation — the OpenCL
# kernel may overwrite its position buffer once it starts).  Collimated along
# +z, so x and y at launch are also x and y at the lens front face.
r0 = np.hypot(src._photons._positions[:, 0], src._photons._positions[:, 1])

# ---------------------------------------------------------------------------
# 3. Propagate
# ---------------------------------------------------------------------------
logger = EnergyLogger(scene)
src.propagate(scene, logger, showProgress=False)

# ---------------------------------------------------------------------------
# 4. Select meridional rays, balanced across entrance radius
# ---------------------------------------------------------------------------
# "Meridional" = photons launched near the x-z plane.  Any photon with small
# |y_0| remains close to y=0 throughout (lens refraction in a spherical system
# doesn't couple x and y for an axial object).
y0 = src._photons._positions[:, 1]
meridional = np.where(np.abs(y0) < MERIDIONAL_SLAB)[0]

# Stratified sampling: take 60 rays from each of 20 equal rho_0 bins in
# [0, 10) mm.  Result = ~1200 rays uniform in rho_0 (some bins may run out if
# the meridional slab is narrow, hence `min(60, len(in_bin))`).
rng = np.random.default_rng(0)
rho_bins = np.linspace(0, 10, 21)
chosen = []
for lo, hi in zip(rho_bins[:-1], rho_bins[1:]):
    in_bin = meridional[(r0[meridional] >= lo) & (r0[meridional] < hi)]
    if len(in_bin) == 0:
        continue
    k = min(60, len(in_bin))
    chosen.append(rng.choice(in_bin, k, replace=False))
chosen = np.concatenate(chosen)
print(f"  showing {len(chosen)} meridional rays (|y_0| < {MERIDIONAL_SLAB} mm)")

# ---------------------------------------------------------------------------
# 5. Build per-photon trajectories
# ---------------------------------------------------------------------------
# `traj[pid]` is a list of (z, x) tuples sorted by z.  We seed it with the
# launch point (z = -20) so the ray starts visibly from the source, not from
# the first screen.
traj = {int(i): [(-20.0, src._photons._positions[int(i), 0])] for i in chosen}

factory = PointCloudFactory(logger)
for z_val, lbl in zip(Z_SCREENS, labels):
    # Every photon's crossing into this screen's front face.
    pts = factory.getPointCloud(lbl, f"{lbl}_front").enteringSurfacePoints
    if pts is None or pts.shape[1] < 5:
        continue
    ids = pts[:, 4].astype(np.int64)
    # Filter to just the rays we're drawing.
    sel = np.isin(ids, chosen)
    for i in np.where(sel)[0]:
        traj[int(ids[i])].append((z_val, pts[i, 1]))

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
# 6.2 x 3.2 inches: wide aspect appropriate for a ray diagram that spans ~17 mm
# in z and only ~3 mm in x.
fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
cmap = plt.get_cmap("viridis")

# Draw marginal (large rho_0) rays first so paraxial rays render on top — the
# inner rays carry more physical significance (define the nominal focus).
sorted_pids = sorted(traj.keys(), key=lambda p: -r0[p])
for pid in sorted_pids:
    pts_list = traj[pid]
    if len(pts_list) < 2:
        continue
    arr = np.asarray(pts_list)
    # Polyline is the TRUE trajectory: in vacuum between screens the photon
    # travels in a straight line, so connecting crossing points is exact.
    c = cmap(r0[pid] / 10.0)
    ax.plot(arr[:, 0], arr[:, 1], "-", color=c, lw=0.4, alpha=0.45,
            rasterized=True)

# Reference line at the nominal paraxial focal length.
ax.axvline(LENS_F, color="k", ls=":", lw=0.9, alpha=0.6)
ax.annotate(r"nominal $f$", xy=(LENS_F, 1.35), xytext=(LENS_F - 0.25, 1.35),
            ha="right", va="center", fontsize=9, color="k", alpha=0.85)

# Publication-standard axis labels: "quantity, symbol (unit)".
ax.set_xlabel(r"axial position, $z$ (mm)", fontsize=11)
ax.set_ylabel(r"transverse position, $x$ (mm)", fontsize=11)
ax.set_xlim(85, 102.5)
ax.set_ylim(-1.6, 1.6)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.25)

# Colorbar for rho_0.  `ScalarMappable` is needed because the colours were
# applied manually per polyline (no direct mappable from ax.plot).
sm = mcm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(0, 10))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=25)
cbar.set_label(r"entrance height, $\rho_0$ (mm)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Save PDF (300 dpi, paper) and PNG (200 dpi, quick preview).
out = os.path.dirname(os.path.abspath(__file__))
for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = os.path.join(out, f"fig_paper_caustic.{ext}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save] {path}")

# ---------------------------------------------------------------------------
# 7. Optional 3D visualization
# ---------------------------------------------------------------------------
# Pops Mayavi windows (needs a real display).  Skipped under NO_3D=1 or
# headless CI environments.
if os.environ.get("NO_3D") != "1":
    from pytissueoptics import Viewer, PointCloudStyle
    print("\n[3D] Showing setup (close window to continue) ...")
    scene.show(source=src)
    print("[3D] Showing post-propagation photon cloud (close window to exit) ...")
    Viewer(scene, src, logger).show3D(
        pointCloudStyle=PointCloudStyle(showSolidPoints=False))
