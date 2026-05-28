"""Figure 3 — 4f relay system (two symmetric lenses, unit magnification).

Publication-figure script.  Runs TWO Monte-Carlo simulations:

  1. On-axis divergent source at z = -f of Lens 1.
     → measures image-plane RMS spot radius and R90 (image quality).
     → produces the two paper figures (intensity profiles, encircled energy).

  2. Off-axis source (same geometry, +1 mm lateral offset).
     → measures image-plane centroid, from which transverse magnification
       M = x_image / x_source is derived.  A perfect 4f relay gives M = -1.

Outputs (separate files, both PDF and PNG, saved alongside this script):

  fig3_4f_intensity.{pdf,png}   — normalized radial intensity profiles
                                  at the Fourier plane and the image plane.
  fig3_4f_encircled.{pdf,png}   — encircled-energy curves at both planes.

3D views:
  Two Mayavi windows pop up during the on-axis run — a SETUP view (scene +
  source, before propagation) and a POST-PROPAGATION view with the photon
  cloud.  Save these manually using Mayavi's built-in save button; the
  script will continue once you close each window.  Skip both with NO_3D=1.

Environment variables:
  N     : photon count for the on-axis (main) run.  Default: 1e6 on GPU.
  N_MAG : photon count for the off-axis magnification run.  Default: 2e5.
  NO_3D : set to 1 to skip both 3D windows (needed for headless runs).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

from pytissueoptics import (
    Cuboid, DivergentSource, EnergyLogger, PointCloudStyle,
    ScatteringMaterial, ScatteringScene, SymmetricLens, Vector, Viewer,
    hardwareAccelerationIsAvailable,
)
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# Main (on-axis) run: 1M photons matches the paper text.  Off-axis (N_MAG)
# only needs a centroid measurement, which converges at ~200k photons.
N_MAIN = int(os.environ.get("N", 1_000_000 if hardwareAccelerationIsAvailable() else 5_000))
N_MAG = int(os.environ.get("N_MAG", 200_000 if hardwareAccelerationIsAvailable() else 2_000))
# Ray-diagram run needs fewer photons since only ~200 rays are plotted.
N_RAY = int(os.environ.get("N_RAY", 50_000 if hardwareAccelerationIsAvailable() else 2_000))
SEED = 42

# Lens geometry (same f, d, t, n for both elements — the "identical" part
# of the claim in the paper caption).
LENS_F = 50.0     # mm — focal length of each lens
LENS_D = 25.4     # mm — clear aperture diameter (1-inch standard)
LENS_T = 3.6      # mm — lens centre thickness
GLASS_N = 1.50    # refractive index

# Divergent source parameters — a small disk with a wide emission cone to
# emulate a fiber-coupled or LED-like point source.
SOURCE_DIAM = 0.1   # mm — source core diameter
SOURCE_DIV = 0.3    # rad — half-angle divergence (~17°)

# Lateral offset for the magnification test.  1 mm is well inside the lens
# aperture, small enough to stay paraxial and large enough to give a high-
# SNR centroid measurement of the image shift.
MAG_OFFSET = 1.0    # mm


# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------
def build_scene(source_offset_x=0.0, N=None, divergence=None):
    """Assemble the 4f scene and source.

    Geometry along the optical axis z:
        source    at z = -f      (+ optional lateral x-offset for mag test)
        Lens 1    at z = 0
        Fourier plane screen at z = +f
        Lens 2    at z = +2f
        Image plane screen at z = +3f      (front face exactly at the BFL)

    Parameters
    ----------
    source_offset_x : float
        Lateral shift of the source along x (mm).  Non-zero for the
        magnification test.
    N : int
        Photon count for this simulation.
    divergence : float or None
        Source divergence (rad, full-cone angle).  None means use the default
        `SOURCE_DIV` value; override to compare imaging at different beam
        radii at the lens aperture (used by the SA-confirmation run).
    """
    if divergence is None:
        divergence = SOURCE_DIV
    glass = ScatteringMaterial(n=GLASS_N)
    vacuum = ScatteringMaterial()

    lens1 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 0), label="Lens1")
    lens2 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 2 * LENS_F),
                          label="Lens2")

    # Thin vacuum cuboids as intensity monitors.  The centre of each cuboid
    # sits at (z + 0.05), so its FRONT face lies exactly on the nominal
    # focal/Fourier plane (simplifies "at-plane" interpretation).
    screen_f = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, LENS_F),
                      material=vacuum, label="FourierPlane")
    screen_i = Cuboid(a=40, b=40, c=0.1, position=Vector(0, 0, 3 * LENS_F + 0.05),
                      material=vacuum, label="ImagePlane")

    scene = ScatteringScene([lens1, lens2, screen_f, screen_i])
    source = DivergentSource(
        position=Vector(source_offset_x, 0, -LENS_F),
        direction=Vector(0, 0, 1),
        N=N,
        diameter=SOURCE_DIAM,
        divergence=divergence,
        displaySize=2,
        seed=SEED,
    )
    return scene, source


def get_screen_hits(logger, screen_label):
    """Return (x, y, w) arrays for photons entering `screen_label`'s front face."""
    pc = PointCloudFactory(logger).getPointCloud(screen_label, f"{screen_label}_front")
    pts = pc.enteringSurfacePoints
    if pts is None or len(pts) == 0:
        return None, None, None
    return pts[:, 1], pts[:, 2], np.abs(pts[:, 0])


def spot_statistics(x, y, w):
    """Compute centroid (x0, y0), RMS spot radius about centroid, and R90.

    The RMS is taken about the energy-weighted centroid — this makes the
    metric meaningful for off-axis images too, not just on-axis ones.
    """
    total = w.sum()
    x0 = (w * x).sum() / total
    y0 = (w * y).sum() / total
    r = np.hypot(x - x0, y - y0)
    rms = float(np.sqrt((w * r ** 2).sum() / total))
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / total
    idx = int(np.searchsorted(cum, 0.90))
    r90 = float(r[order][min(idx, len(r) - 1)])
    return (float(x0), float(y0)), rms, r90


def radial_intensity_profile(x, y, w, n_bins=70, r_min=3e-2, r_max=15.0,
                              smooth_window=5):
    """Energy per unit area vs radius on a log-spaced r grid.

    Why log-spaced bins: the image-plane spot is ~0.3 mm wide while the
    Fourier-plane beam is ~14 mm wide — two orders of magnitude apart in
    characteristic scale.  A linear-r grid wastes bins in one regime or the
    other; log-r resolves both on the same figure.

    Why a running-mean smooth: the smallest annular bins have very small area
    (pi*(r_hi^2 - r_lo^2)) and therefore few photons, so the per-area density
    is noisy for r ~ r_min.  A short running mean in log-r removes that shot
    noise without distorting the flat-top/spot features we care about.

    Returns (r_centers, I / I.max()).
    """
    r = np.hypot(x, y)
    edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    hist, _ = np.histogram(r, bins=edges, weights=w)
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    centers = np.sqrt(edges[:-1] * edges[1:])       # geometric-mean bin centre
    intensity = hist / areas
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        intensity = np.convolve(intensity, kernel, mode="same")
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    return centers, intensity


def encircled_energy(x, y, w):
    """Cumulative fraction of energy enclosed within radius r from origin."""
    r = np.hypot(x, y)
    order = np.argsort(r)
    cum = np.cumsum(w[order]) / w.sum()
    return r[order], cum


# ---------------------------------------------------------------------------
# 3. Simulation 1 — on-axis source
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SIMULATION 1: on-axis source")
print("=" * 70)

scene, source = build_scene(source_offset_x=0.0, N=N_MAIN)

# 3D geometry preview (user saves manually from the Mayavi window).
if os.environ.get("NO_3D") != "1":
    print("\n[3D] Showing setup (close window to continue) ...")
    scene.show(source=source)

logger = EnergyLogger(scene)
source.propagate(scene, logger)

# 3D post-propagation photon cloud (user saves manually).
if os.environ.get("NO_3D") != "1":
    print("[3D] Showing post-propagation photon cloud (close window to continue) ...")
    Viewer(scene, source, logger).show3D(
        pointCloudStyle=PointCloudStyle(showSolidPoints=False))

# Measure RMS + R90 at both monitor planes.
x_f, y_f, w_f = get_screen_hits(logger, "FourierPlane")
x_i, y_i, w_i = get_screen_hits(logger, "ImagePlane")
(cx_f, cy_f), rms_f, r90_f = spot_statistics(x_f, y_f, w_f)
(cx_i, cy_i), rms_i, r90_i = spot_statistics(x_i, y_i, w_i)

# ---------------------------------------------------------------------------
# 4. Simulation 2 — off-axis source for magnification
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SIMULATION 2: off-axis source (magnification test)")
print("=" * 70)
scene2, source2 = build_scene(source_offset_x=MAG_OFFSET, N=N_MAG)
logger2 = EnergyLogger(scene2)
source2.propagate(scene2, logger2)

x_i2, y_i2, w_i2 = get_screen_hits(logger2, "ImagePlane")
(cx_i2, cy_i2), _, _ = spot_statistics(x_i2, y_i2, w_i2)

# Transverse magnification M = x_image / x_source.
# For an ideal 4f relay the expected value is -1 (inverted unit magnification).
M = cx_i2 / MAG_OFFSET

# ---------------------------------------------------------------------------
# 4aa. Simulation 2b — M(field) sweep
# ---------------------------------------------------------------------------
# A single-point magnification measurement could be fortuitous.  Sweeping the
# source across ±3 mm and measuring the image centroid at each position lets
# us verify that M is CONSTANT over the field of view (= field-independent
# magnification = no imaging distortion), which is the proper referee-grade
# statement of "unit magnification".
print("\n" + "=" * 70)
print("SIMULATION 2b: M(field) sweep")
print("=" * 70)
FIELD_OFFSETS = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]  # mm
field_image_x = []
for dx in FIELD_OFFSETS:
    scene_f, src_f = build_scene(source_offset_x=dx, N=N_MAG)
    logger_f = EnergyLogger(scene_f)
    src_f.propagate(scene_f, logger_f, showProgress=False)
    xf, yf, wf = get_screen_hits(logger_f, "ImagePlane")
    (cx_f_im, _), _, _ = spot_statistics(xf, yf, wf)
    field_image_x.append(cx_f_im)
    print(f"  source x = {dx:+.2f} mm  →  image x = {cx_f_im:+.4f} mm  →"
          f"  M = {cx_f_im / dx:+.4f}")

# Fit image_x = M_fit * source_x by least squares (forcing through origin is
# fine since on-axis source → on-axis image by symmetry).
field_offsets_arr = np.asarray(FIELD_OFFSETS)
field_image_arr = np.asarray(field_image_x)
M_fit = float((field_offsets_arr * field_image_arr).sum()
              / (field_offsets_arr ** 2).sum())
print(f"  Least-squares slope over all 8 points: M_fit = {M_fit:+.4f}")

# ---------------------------------------------------------------------------
# 4ab. Simulation 2c — spherical-aberration confirmation via best-focus scan
# ---------------------------------------------------------------------------
# Why a scan instead of a single-plane measurement:
# The nominal image plane at z = 3f = 150 mm is computed from the thin-lens
# approximation.  For a thick biconvex (t = 3.6 mm, n = 1.5) the true front
# principal plane lies ~1 mm inside the glass, so placing the source at
# z = -f introduces a ~0.8 mm defocus offset.  Measuring RMS at a single z
# mixes SA with defocus: at wide divergence SA dominates, at narrow
# divergence defocus dominates, giving the misleading result that narrowing
# the cone INCREASES the RMS at z = 150.
#
# The clean aberration measurement is the MINIMUM RMS across a small axial
# scan centred on the thin-lens image plane: that minimum is the
# aberration-limited spot size, independent of the thick-lens defocus
# offset.  Primary SA scales as ρ² so the SA-limited minimum should drop
# significantly when the beam is narrowed.
print("\n" + "=" * 70)
print("SIMULATION 2c: SA confirmation (best-focus RMS scan vs divergence)")
print("=" * 70)

IMAGE_SCAN_Z = np.array([149.0, 149.5, 150.0, 150.5, 151.0, 151.5, 152.0])
DIV_VALUES = (0.10, 0.30)    # rad — narrow vs wide beam

def _scan_build(divergence):
    """Build a scene identical to build_scene() but with a z-scan of
    image-plane screens replacing the single ImagePlane cuboid.  We keep
    the FourierPlane screen too in case it is needed downstream."""
    glass = ScatteringMaterial(n=GLASS_N)
    vacuum = ScatteringMaterial()
    lens1 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 0), label="Lens1")
    lens2 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 2 * LENS_F),
                          label="Lens2")
    scan_screens = [
        Cuboid(a=40, b=40, c=0.05, position=Vector(0, 0, z + 0.025),
               material=vacuum, label=f"scan{z:.2f}")
        for z in IMAGE_SCAN_Z
    ]
    scene = ScatteringScene([lens1, lens2, *scan_screens])
    source = DivergentSource(
        position=Vector(0, 0, -LENS_F), direction=Vector(0, 0, 1), N=N_MAIN,
        diameter=SOURCE_DIAM, divergence=divergence,
        displaySize=2, seed=SEED,
    )
    return scene, source


sa_best = {}   # divergence -> (z_best, rms_best)
for div in DIV_VALUES:
    scene_s, src_s = _scan_build(div)
    logger_s = EnergyLogger(scene_s)
    src_s.propagate(scene_s, logger_s, showProgress=False)
    rms_profile = np.full(len(IMAGE_SCAN_Z), np.nan)
    for iz, z in enumerate(IMAGE_SCAN_Z):
        xs, ys, ws = get_screen_hits(logger_s, f"scan{z:.2f}")
        if xs is None:
            continue
        (_, _), rms_profile[iz], _ = spot_statistics(xs, ys, ws)
    imin = int(np.nanargmin(rms_profile))
    sa_best[div] = (float(IMAGE_SCAN_Z[imin]), float(rms_profile[imin]))
    print(f"  div = {div:.2f} rad  :  best-focus at z = {IMAGE_SCAN_Z[imin]:.2f} mm, "
          f"min RMS = {rms_profile[imin]:.3f} mm")

# Predicted ρ² scaling of the SA-limited spot:
r_wide = LENS_F * np.tan(SOURCE_DIV / 2)
r_narrow = LENS_F * np.tan(0.10 / 2)
rms_wide_best = sa_best[0.30][1]
rms_narrow_best = sa_best[0.10][1]
print(f"  beam radius at lens: wide = {r_wide:.1f} mm, narrow = {r_narrow:.1f} mm")
print(f"  best-focus RMS reduction: {rms_wide_best / rms_narrow_best:.1f}x     "
      f"(ρ² prediction: {(r_wide / r_narrow) ** 2:.1f}x)")

# ---------------------------------------------------------------------------
# 4b. Simulation 3 — dense-screen run for the ray diagram
# ---------------------------------------------------------------------------
# Visual proof that the 4f relay works: we launch 50k photons from the source
# and log their crossings at ~20 screens spanning the entire propagation path
# (source → L1 → Fourier → L2 → image).  Connecting each photon's crossings by
# straight-line segments gives its true trajectory (vacuum between screens →
# rays are exactly linear between logging planes).  The resulting polyline
# figure shows at a glance: divergent cone from source, parallel slab between
# lenses, convergent cone onto the image plane.
print("\n" + "=" * 70)
print("SIMULATION 3: dense-screen run for the ray diagram")
print("=" * 70)

# Dense screen grid spanning z = -45 mm (near source) to z = 150 mm (image
# plane).  Each z is chosen to avoid overlap with the lens bodies (lens 1
# occupies z ∈ [-1.8, 1.8], lens 2 z ∈ [98.2, 101.8]).
Z_RAY = np.array([-45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65,
                   75, 85, 95, 105, 115, 125, 135, 145, 150])


def build_ray_scene():
    """Scene dedicated to the ray diagram: same lenses, many logging screens."""
    glass = ScatteringMaterial(n=GLASS_N)
    vacuum = ScatteringMaterial()
    lens1 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 0), label="Lens1")
    lens2 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 2 * LENS_F),
                          label="Lens2")
    screens = [
        Cuboid(40, 40, 0.05, position=Vector(0, 0, z + 0.025),
               material=vacuum, label=f"r{z:+.1f}")
        for z in Z_RAY
    ]
    scene = ScatteringScene([lens1, lens2, *screens])
    source = DivergentSource(
        position=Vector(0, 0, -LENS_F), direction=Vector(0, 0, 1), N=N_RAY,
        diameter=SOURCE_DIAM, divergence=SOURCE_DIV, displaySize=2, seed=SEED,
    )
    return scene, source


scene3, source3 = build_ray_scene()
# Read launch state BEFORE propagation: launch position (x0, y0, -f) and
# direction (dx0, dy0, dz0).  dx0 ≈ sin(θ_x) for small angles = emission angle.
x0_launch = source3._photons._positions[:, 0]
y0_launch = source3._photons._positions[:, 1]
dx0_launch = source3._photons._directions[:, 0]
dy0_launch = source3._photons._directions[:, 1]

logger3 = EnergyLogger(scene3)
source3.propagate(scene3, logger3, showProgress=False)

# Select meridional rays (small |y₀|, small |dy₀|) so projecting onto the
# (z, x) plane faithfully represents the ray path.
meridional = np.where((np.abs(y0_launch) < 0.04) &
                       (np.abs(dy0_launch) < 0.04))[0]

# Stratified sample across launch angle dx0 so every colour is represented.
rng = np.random.default_rng(1)
dx_bins = np.linspace(-0.16, 0.16, 17)          # 16 bins over ±0.15 rad cone
chosen = []
for lo, hi in zip(dx_bins[:-1], dx_bins[1:]):
    in_bin = meridional[(dx0_launch[meridional] >= lo) &
                         (dx0_launch[meridional] < hi)]
    if len(in_bin) == 0:
        continue
    chosen.append(rng.choice(in_bin, min(15, len(in_bin)), replace=False))
chosen = np.concatenate(chosen)
print(f"  selected {len(chosen)} meridional rays from {N_RAY:,} launched")

# Build polylines: {pid: [(z, x), ...]} starting at the source (z = -f).
traj = {int(pid): [(-LENS_F, x0_launch[pid])] for pid in chosen}
factory3 = PointCloudFactory(logger3)
for z_val in Z_RAY:
    lbl = f"r{z_val:+.1f}"
    pts = factory3.getPointCloud(lbl, f"{lbl}_front").enteringSurfacePoints
    if pts is None or pts.shape[1] < 5:
        continue
    ids = pts[:, 4].astype(np.int64)
    sel = np.isin(ids, chosen)
    for i in np.where(sel)[0]:
        traj[int(ids[i])].append((z_val, pts[i, 1]))

# ---------------------------------------------------------------------------
# 5. Print summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Photons (on-axis run):        {N_MAIN:,}")
print(f"  Photons (off-axis run):       {N_MAG:,}")
print(f"\n  Fourier plane:")
print(f"    centroid = ({cx_f:+.3f}, {cy_f:+.3f}) mm")
print(f"    RMS spot radius = {rms_f:.3f} mm")
print(f"    R90 = {r90_f:.3f} mm")
print(f"\n  Image plane (on-axis source):")
print(f"    centroid = ({cx_i:+.3f}, {cy_i:+.3f}) mm")
print(f"    RMS spot radius = {rms_i:.3f} mm")
print(f"    R90 = {r90_i:.3f} mm")
print(f"\n  Magnification test (source offset = {MAG_OFFSET:+.3f} mm in x):")
print(f"    image centroid = ({cx_i2:+.3f}, {cy_i2:+.3f}) mm")
print(f"    transverse magnification M = {M:+.4f}")
print(f"    |M| = {abs(M):.4f}   (expected 1.000 for 4f relay, sign -1)")
print(f"\n  M(field) sweep (±3 mm):")
print(f"    least-squares slope M_fit = {M_fit:+.4f}")
print(f"\n  SA confirmation (best-focus RMS scan):")
print(f"    wide   (div = 0.30 rad, r_lens = {r_wide:.1f} mm): "
      f"z_best = {sa_best[0.30][0]:.2f} mm, RMS_min = {rms_wide_best:.3f} mm")
print(f"    narrow (div = 0.10 rad, r_lens = {r_narrow:.1f} mm): "
      f"z_best = {sa_best[0.10][0]:.2f} mm, RMS_min = {rms_narrow_best:.3f} mm")
print(f"    best-focus RMS reduction = {rms_wide_best / rms_narrow_best:.1f}x, "
      f"ρ² prediction {(r_wide / r_narrow) ** 2:.1f}x")

# PASS/FAIL against the paper's quoted numbers.
rms_expected = 0.38          # mm  (paper text)
rms_tol = 0.05               # mm
M_expected = -1.0
M_tol = 0.05
rms_ok = abs(rms_i - rms_expected) < rms_tol
M_ok = abs(M - M_expected) < M_tol
print()
print(f"  [{ 'PASS' if rms_ok else 'FAIL'}] image RMS within ±{rms_tol} mm of {rms_expected} mm"
      f"  (measured {rms_i:.3f} mm)")
print(f"  [{ 'PASS' if M_ok else 'FAIL'}] magnification within ±{M_tol} of {M_expected}"
      f"           (measured {M:+.4f})")
# Additional PASS/FAIL: M(field) linearity and SA confirmation.
M_field_ok = abs(M_fit - M_expected) < M_tol
sa_reduction = rms_wide_best / rms_narrow_best
sa_predicted = (r_wide / r_narrow) ** 2
sa_ok = 0.4 < sa_reduction / sa_predicted < 2.5   # factor 2 envelope around ρ² scaling
print(f"  [{ 'PASS' if M_field_ok else 'FAIL'}] M constant over ±3 mm field"
      f"              (M_fit = {M_fit:+.4f})")
print(f"  [{ 'PASS' if sa_ok else 'FAIL'}] SA ρ² scaling: RMS reduction "
      f"{sa_reduction:.1f}x vs predicted {sa_predicted:.1f}x")

# ---------------------------------------------------------------------------
# 6. Save the two paper figures SEPARATELY
# ---------------------------------------------------------------------------
out_dir = os.path.dirname(os.path.abspath(__file__))


def save_fig(fig, basename):
    for ext, dpi in [("pdf", 300), ("png", 200)]:
        path = os.path.join(out_dir, f"{basename}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {path}")


# 6a) Radial intensity profile at both planes (log-log: the image-plane spike
#     at r ~ 0.3 mm and the Fourier-plane flat-top out to r ~ 7 mm would
#     otherwise require incompatible linear scales).  Legend placed OUTSIDE
#     the plot box so it never overlaps the data curves.
fig_int, ax = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
for name, (x, y, w), color in [
    ("Fourier plane", (x_f, y_f, w_f), "#1f77b4"),
    ("image plane",   (x_i, y_i, w_i), "#d62728"),
]:
    r, I = radial_intensity_profile(x, y, w)
    ax.plot(r, I, color=color, lw=1.6, label=name)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"radial distance, $r$ (mm)", fontsize=11)
ax.set_ylabel("normalized intensity (a.u.)", fontsize=11)
ax.set_xlim(3e-2, 15)
ax.set_ylim(1e-5, 2)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False, handletextpad=0.4, borderpad=0.3)
save_fig(fig_int, "fig3_4f_intensity")

# 6b) Encircled energy curves.  Log-x so both the image-plane (saturates near
#     0.5 mm) and the Fourier-plane (saturates near 10 mm) curves are visible.
#     Legend also placed outside the plot box.
fig_ee, ax = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
for name, (x, y, w), color, r90 in [
    ("Fourier plane", (x_f, y_f, w_f), "#1f77b4", r90_f),
    ("image plane",   (x_i, y_i, w_i), "#d62728", r90_i),
]:
    r, ee = encircled_energy(x, y, w)
    ax.plot(r, ee * 100, color=color, lw=1.6,
            label=f"{name}\n($R_{{90}} = {r90:.2f}$ mm)")
    # Mark the R90 point on each curve.
    ax.plot(r90, 90, marker="o", color=color, ms=5, zorder=10)
ax.axhline(90, color="gray", ls=":", lw=0.8, alpha=0.7)
ax.set_xscale("log")
ax.set_xlabel(r"radial distance, $r$ (mm)", fontsize=11)
ax.set_ylabel("encircled energy (%)", fontsize=11)
ax.set_xlim(1e-2, 15)
ax.set_ylim(0, 105)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False, handletextpad=0.4, borderpad=0.3)
save_fig(fig_ee, "fig3_4f_encircled")

# 6c) Ray-diagram figure: continuous polylines through the whole 4f path.
# Colour each ray by its emission angle dx0 so the reader can visually track a
# given ray from source through Fourier plane to image.  Shaded vertical
# strips mark the two lens bodies; dotted verticals mark the source plane,
# Fourier plane, and image plane.
fig_r, ax = plt.subplots(figsize=(7.0, 3.0), constrained_layout=True)

# Lens bodies (shaded grey strips)
ax.axvspan(-LENS_T / 2, LENS_T / 2, color="0.4", alpha=0.25, zorder=0)
ax.axvspan(2 * LENS_F - LENS_T / 2, 2 * LENS_F + LENS_T / 2,
           color="0.4", alpha=0.25, zorder=0)

# Key planes (dotted verticals)
for z_plane in (-LENS_F, LENS_F, 3 * LENS_F):
    ax.axvline(z_plane, color="k", ls=":", lw=0.7, alpha=0.5, zorder=0)

# Plot rays, sorted so that extreme angles render on top.
cmap_ray = plt.get_cmap("coolwarm")
chosen_sorted = sorted(chosen, key=lambda p: abs(dx0_launch[p]))
for pid in chosen_sorted:
    pts_list = traj[int(pid)]
    if len(pts_list) < 2:
        continue
    arr = np.asarray(pts_list)
    # Map dx0 in [-0.16, +0.16] to colormap [0, 1].
    c = cmap_ray(0.5 + dx0_launch[pid] / 0.32)
    ax.plot(arr[:, 0], arr[:, 1], "-", color=c, lw=0.4, alpha=0.55,
            rasterized=True)

# Labels placed OUTSIDE the axes (just above the top spine) using
# xaxis_transform — y is axes-fraction, x is in data coords.  This guarantees
# the labels never overlap the rays regardless of what r-range the data span.
xtf = ax.get_xaxis_transform()
ax.text(0, 1.03, "L1", ha="center", va="bottom", fontsize=9, weight="bold",
        transform=xtf)
ax.text(2 * LENS_F, 1.03, "L2", ha="center", va="bottom", fontsize=9,
        weight="bold", transform=xtf)
ax.text(-LENS_F, 1.03, "source", ha="center", va="bottom", fontsize=8,
        transform=xtf)
ax.text(LENS_F, 1.03, "Fourier", ha="center", va="bottom", fontsize=8,
        transform=xtf)
ax.text(3 * LENS_F, 1.03, "image", ha="center", va="bottom", fontsize=8,
        transform=xtf)

ax.set_xlabel(r"axial position, $z$ (mm)", fontsize=11)
ax.set_ylabel(r"transverse position, $x$ (mm)", fontsize=11)
ax.set_xlim(-55, 160)
ax.set_ylim(-13, 13)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.2)

# Colorbar: launch angle.
sm_ray = mcm.ScalarMappable(cmap="coolwarm",
                             norm=mcolors.Normalize(-0.15, 0.15))
sm_ray.set_array([])
cbar = fig_r.colorbar(sm_ray, ax=ax, pad=0.015, aspect=20)
cbar.set_label(r"launch angle, $\theta_x$ (rad)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

save_fig(fig_r, "fig3_4f_rays")

# 6d) Beam-radius profile along the optical axis, R90(z), from the same
# dense-screen sim.  The characteristic 4f signature is: small waist at the
# source plane, linear divergence, flat plateau at the aperture-limited beam
# radius between the lenses, linear convergence after L2, minimum at the
# image plane.  Any departure from this shape indicates the relay is
# misaligned.
r90_z = np.full(len(Z_RAY), np.nan)
factory_r = PointCloudFactory(logger3)
for iz, z in enumerate(Z_RAY):
    lbl = f"r{z:+.1f}"
    pts = factory_r.getPointCloud(lbl, f"{lbl}_front").enteringSurfacePoints
    if pts is None or len(pts) == 0:
        continue
    r = np.hypot(pts[:, 1], pts[:, 2])
    w = np.abs(pts[:, 0])
    order = np.argsort(r)
    cum = np.cumsum(w[order])
    idx = int(np.searchsorted(cum, 0.9 * cum[-1]))
    r90_z[iz] = r[order][min(idx, len(r) - 1)]

fig_w, ax = plt.subplots(figsize=(6.2, 3.0), constrained_layout=True)
ax.plot(Z_RAY, r90_z, "-o", ms=4, lw=1.5, color="#1f77b4")

# Shade the lens bodies and mark the key planes.
ax.axvspan(-LENS_T / 2, LENS_T / 2, color="0.4", alpha=0.25, zorder=0)
ax.axvspan(2 * LENS_F - LENS_T / 2, 2 * LENS_F + LENS_T / 2,
           color="0.4", alpha=0.25, zorder=0)
for z_plane in (-LENS_F, LENS_F, 3 * LENS_F):
    ax.axvline(z_plane, color="k", ls=":", lw=0.7, alpha=0.5, zorder=0)

# Key-plane labels outside the axes (same trick as the ray diagram).
xtf = ax.get_xaxis_transform()
ax.text(0, 1.03, "L1", ha="center", va="bottom", fontsize=9, weight="bold",
        transform=xtf)
ax.text(2 * LENS_F, 1.03, "L2", ha="center", va="bottom", fontsize=9,
        weight="bold", transform=xtf)
ax.text(-LENS_F, 1.03, "source", ha="center", va="bottom", fontsize=8,
        transform=xtf)
ax.text(LENS_F, 1.03, "Fourier", ha="center", va="bottom", fontsize=8,
        transform=xtf)
ax.text(3 * LENS_F, 1.03, "image", ha="center", va="bottom", fontsize=8,
        transform=xtf)

ax.set_xlabel(r"axial position, $z$ (mm)", fontsize=11)
ax.set_ylabel(r"$R_{90}$ beam radius (mm)", fontsize=11)
ax.set_yscale("log")
ax.set_xlim(-55, 160)
ax.set_ylim(5e-2, 20)
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.3, which="both")
save_fig(fig_w, "fig3_4f_waist")

# 6e) Magnification field linearity plot: image centroid vs source offset.
# For a perfect 4f the points lie exactly on y = -x (slope -1, through origin).
fig_m, ax = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
ax.axhline(0, color="0.8", lw=0.5, zorder=0)
ax.axvline(0, color="0.8", lw=0.5, zorder=0)
# Theoretical line M = -1
x_line = np.linspace(-3.5, 3.5, 2)
ax.plot(x_line, -x_line, color="gray", ls="--", lw=1.0,
        label=r"ideal $M = -1$", zorder=1)
# Measured points
ax.plot(field_offsets_arr, field_image_arr, "o", color="#d62728", ms=5,
        label=rf"measured ($M_{{\mathrm{{fit}}}} = {M_fit:+.4f}$)", zorder=2)
ax.set_xlabel(r"source offset, $x_{\mathrm{src}}$ (mm)", fontsize=11)
ax.set_ylabel(r"image centroid, $x_{\mathrm{img}}$ (mm)", fontsize=11)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect("equal")
ax.tick_params(axis="both", labelsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=9, loc="upper right", frameon=False)
save_fig(fig_m, "fig3_4f_magnification")
