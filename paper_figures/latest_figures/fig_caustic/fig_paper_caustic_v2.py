"""Publication figure (v2): meridional caustic of a biconvex thick lens.

Same physics as fig_paper_caustic.py, but the rays are drawn from NATIVE photon
traces rather than the 19-screen scaffold, plus styling fixes (no grid, thicker
dashed nominal-f reference, capitalized axis labels).

Native photon-trace paths
-------------------------
PyTissueOptics logs a photon's (x, y, z) only where it crosses a surface, and in
this transparent scene there are no interactions between the lens and the far
field -- so each photon natively logs just its lens front/back crossings. Past
the back surface the ray is a single STRAIGHT line, so two points fix it exactly.
We therefore keep ONE downstream capture plane (z = 105 mm, off the plotted
range) and, for each photon, read its lens-back exit and that capture crossing
straight from the logger's raw events (grouped by photonID). The post-lens ray is
the line through those two real crossings; we draw it across the focal window.
This is geometrically exact (no interpolation) and uses a single capture surface
instead of the 19-screen stack.

Environment variables
---------------------
N       : number of photons launched (default 500k on GPU).
NO_3D=1 : skip interactive 3D viewers at the end.
"""
import os

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from pytissueoptics import (
    Cuboid, DirectionalSource, EnergyLogger, ScatteringMaterial,
    ScatteringScene, SymmetricLens, Vector, hardwareAccelerationIsAvailable,
)

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
N = int(os.environ.get("N", 500_000 if hardwareAccelerationIsAvailable() else 5_000))
SEED = 42

LENS_F, LENS_D, LENS_T, GLASS_N, BEAM_D = 100.0, 25.4, 3.6, 1.50, 20.0
MERIDIONAL_SLAB = 0.5   # mm; keep only photons launched with |y_0| < this

# Single downstream capture plane. It sits past the plotted window (z up to 102.5)
# so it never appears in the figure; it exists only to give each ray a second
# post-lens point, from which the exact straight trajectory is reconstructed.
Z_CAPTURE = 105.0

PLOT_Z0, PLOT_Z1 = 85.0, 102.5   # axial window shown in the figure

# ---------------------------------------------------------------------------
# 2. Build the scene  (lens + ONE capture plane)
# ---------------------------------------------------------------------------
glass, vacuum = ScatteringMaterial(n=GLASS_N), ScatteringMaterial()
lens = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                     material=glass, position=Vector(0, 0, 0))
capture = Cuboid(40, 40, 0.05, position=Vector(0, 0, Z_CAPTURE + 0.025),
                 material=vacuum, label="capture")
scene = ScatteringScene([lens, capture])

src = DirectionalSource(Vector(0, 0, -20), Vector(0, 0, 1), diameter=BEAM_D,
                        N=N, displaySize=5, seed=SEED)

# Entrance-pupil radius per photon (collimated, so launch x,y == x,y at the lens).
r0 = np.hypot(src._photons._positions[:, 0], src._photons._positions[:, 1])
y0 = src._photons._positions[:, 1]
x0_launch = src._photons._positions[:, 0]

# ---------------------------------------------------------------------------
# 3. Propagate
# ---------------------------------------------------------------------------
logger = EnergyLogger(scene)
src.propagate(scene, logger, showProgress=False)

# ---------------------------------------------------------------------------
# 4. Select meridional rays, balanced across entrance radius
# ---------------------------------------------------------------------------
meridional = np.where(np.abs(y0) < MERIDIONAL_SLAB)[0]
rng = np.random.default_rng(0)
rho_bins = np.linspace(0, 10, 21)
chosen = []
for lo, hi in zip(rho_bins[:-1], rho_bins[1:]):
    in_bin = meridional[(r0[meridional] >= lo) & (r0[meridional] < hi)]
    if len(in_bin) == 0:
        continue
    chosen.append(rng.choice(in_bin, min(60, len(in_bin)), replace=False))
chosen = np.concatenate(chosen)
chosen_set = set(int(i) for i in chosen)
print(f"  showing {len(chosen)} meridional rays (|y_0| < {MERIDIONAL_SLAB} mm)")

# ---------------------------------------------------------------------------
# 5. Reconstruct each ray's post-lens straight line from native events
# ---------------------------------------------------------------------------
# Group the raw logged events (value, x, y, z, photonID) by photonID. For each
# chosen photon, the lens-back exit (largest z in the lens region) and the capture
# crossing (z > 50) are two real points on the straight post-lens ray.
raw = logger.getRawDataPoints()
ids = raw[:, 4].astype(np.int64)
order = np.argsort(ids, kind="stable")
raw, ids = raw[order], ids[order]
uniq, start = np.unique(ids, return_index=True)
bounds = np.append(start, len(ids))
id_to_k = {int(u): k for k, u in enumerate(uniq)}

rays = {}   # pid -> (x_at_Z0, x_at_Z1)
for pid in chosen_set:
    k = id_to_k.get(pid)
    if k is None:
        continue
    ev = raw[bounds[k]:bounds[k + 1]]
    z = ev[:, 3]
    pre, post = ev[z < 50.0], ev[z > 50.0]
    if len(pre) == 0 or len(post) == 0:
        continue
    back = pre[np.argmax(pre[:, 3])]          # lens back exit
    cap = post[np.argmin(post[:, 3])]         # capture-plane crossing
    zb, xb, zc, xc = back[3], back[1], cap[3], cap[1]
    slope = (xc - xb) / (zc - zb)             # exact (vacuum -> straight line)
    rays[pid] = (xb + slope * (PLOT_Z0 - zb), xb + slope * (PLOT_Z1 - zb))

# ---------------------------------------------------------------------------
# 6. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
cmap = plt.get_cmap("viridis")

# Marginal (large rho_0) rays first so paraxial rays render on top.
for pid in sorted(rays.keys(), key=lambda p: -r0[p]):
    x_lo, x_hi = rays[pid]
    ax.plot([PLOT_Z0, PLOT_Z1], [x_lo, x_hi], "-", color=cmap(r0[pid] / 10.0),
            lw=0.4, alpha=0.45, rasterized=True)

# Nominal paraxial focal length: thick, dashed, high-contrast black reference.
ax.axvline(LENS_F, color="black", ls="--", lw=1.8, alpha=0.9)
ax.annotate(r"nominal $f$", xy=(LENS_F, 1.35), xytext=(LENS_F - 0.25, 1.35),
            ha="right", va="center", fontsize=9, color="black")

ax.set_xlabel(r"Axial position, $z$ (mm)", fontsize=11)
ax.set_ylabel(r"Transverse position, $x$ (mm)", fontsize=11)
ax.set_xlim(PLOT_Z0, PLOT_Z1)
ax.set_ylim(-1.6, 1.6)
ax.tick_params(axis="both", labelsize=10)
ax.grid(False)

sm = mcm.ScalarMappable(cmap="viridis", norm=mcolors.Normalize(0, 10))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=25)
cbar.set_label(r"Entrance height, $\rho_0$ (mm)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

out = os.path.dirname(os.path.abspath(__file__))
for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = os.path.join(out, f"fig_paper_caustic_v2.{ext}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save] {path}")

# ---------------------------------------------------------------------------
# 7. Optional 3D visualization
# ---------------------------------------------------------------------------
if os.environ.get("NO_3D") != "1":
    from pytissueoptics import PointCloudStyle, Viewer
    print("\n[3D] Showing setup (close window to continue) ...")
    scene.show(source=src)
    print("[3D] Showing post-propagation photon cloud (close window to exit) ...")
    Viewer(scene, src, logger).show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))
