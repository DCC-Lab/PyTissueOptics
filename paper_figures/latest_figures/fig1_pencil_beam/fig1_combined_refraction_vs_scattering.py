"""
Figure 1 (combined) – Pencil beam through a three-layer stack: refraction vs scattering.

Builds TWO scenes that share *identical* geometry (the PhantomTissue layout:
w = 3, thicknesses = [0.75, 0.5, 0.75], n = [1.4, 1.7, 1.4], centered at z = 1):

  (a) Refraction  : mu_s = 0  -> pure Snell/Fresnel refraction (verified analytically)
  (b) Scattering  : PhantomTissue (mu_s = [2, 3, 2], mu_a = [1, 1, 2], g = 0.8)

Both 3D panels are rendered with an IDENTICAL pinned Mayavi camera (POV), identical
field of view (camera.view_angle), identical render size, and a white background, so the
two panels register exactly for a side-by-side (a/b) comparison.

The Snell + Fresnel + Beer-Lambert verification is printed to the console and runs on
the non-scattering scene (Snell/Fresnel are only well defined without scattering).

Verification tolerances (all reported with achieved deltas so precision is transparent):
  - PER_PHOTON_ANGLE_TOL = 0.02 deg : per-photon median ray angle vs Snell. This is the
       *authoritative* angular check -- in a non-scattering layer r_exit - r_entry is the
       exact ray, so the distribution collapses onto the Snell angle. 0.02 deg is tight.
  - MEANPOS_ANGLE_TOL    = 0.50 deg : coarse cross-check from mean surface positions only
       (sensitive to the Fresnel-reflected tail); shown for reference, not authoritative.
  - PERCENT_TOL          = 1.00 pp  : Fresnel reflectance / transmission / Beer-Lambert.
"""

import math
import os

import numpy as np

from pytissueoptics import *  # noqa: F403
from pytissueoptics.rayscattering.display.viewer import PointCloudStyle, Viewer
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory
from pytissueoptics.rayscattering.statistics import Stats
from pytissueoptics.scene import ViewPointStyle, get3DViewer

# --------------------------------------------------------------------------------------
# Shared camera / render configuration  (identical for BOTH panels -> registered views)
# --------------------------------------------------------------------------------------
# distance + focalpoint are pinned to explicit numbers (NOT the auto-fit None defaults) so
# the two scenes -- whose point clouds have different extents -- share the exact same camera.
#
# Camera presets (azimuth, elevation) -- the beam travels in the x-z plane (y = 0), so
# looking down the y-axis (azimuth ~ +/-90) gives a side view that reveals the full path:
#   original "optics"   : azimuth=-30,  elevation=215
#   side (slightly below): azimuth=-85, elevation=250   <- default; shows the x-z beam path
#   side (slightly above): azimuth=-85, elevation=70
#   pure side / front    : azimuth=-90, elevation=270
CAMERA = dict(
    azimuth=-85.0,                  # near -90 -> looking ~down the y-axis = side view
    elevation=250.0,                # ~20° below level; keeps the "beam goes down" orientation
    distance=11.0,                  # tune to zoom; same for both panels
    focalpoint=(0.0, 0.0, 1.0),     # geometric center of the layer stack (z = sum(t)/2)
    roll=0.0,                       # if the slab looks rotated in the render, set this (90 / -90)
)
VIEW_ANGLE = 30.0                   # perspective FOV in degrees -- identical for both panels
RENDER_SIZE = (900, 900)           # identical render size -> identical aspect/framing

# --- Colors (white background; the default "rainbow" puts low-energy points in blue/purple,
#     which has poor contrast -- use a warm map with no blue/purple, dark = high energy) ---
BG_COLOR = (1.0, 1.0, 1.0)         # white background
FG_COLOR = (0.0, 0.0, 0.0)         # black foreground (text/axes) on white
BEAM_COLORMAP = "hot"              # photon-energy point cloud (black->red->orange->yellow)
BEAM_REVERSE = True                # high energy -> dark (strong contrast on white); low -> faint
SOLID_COLOR = (0.55, 0.55, 0.60)   # translucent gray slabs
SOLID_OPACITY = 0.12
SOURCE_COLOR = (0.0, 0.50, 0.25)   # green source marker, distinct from the warm beam

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Verification tolerances
PER_PHOTON_ANGLE_TOL = 0.02   # deg  (authoritative)
MEANPOS_ANGLE_TOL = 0.50      # deg  (coarse cross-check)
PERCENT_TOL = 1.00            # percentage points

# Shared source / incidence
ANGLE_DEG = 30
SOURCE_POSITION = (-0.5, 0, -0.3)
SOURCE_DISPLAY_SIZE = 0.3


# ======================================================================================
#  Analytic verification (Snell / Fresnel / Beer-Lambert) -- non-scattering scene only
# ======================================================================================
def fresnel_unpolarized(n1, n2, theta_i):
    """Fresnel reflection coefficient for unpolarized light + transmitted angle."""
    theta_t = math.asin((n1 / n2) * math.sin(theta_i))
    cos_i, cos_t = math.cos(theta_i), math.cos(theta_t)
    Rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    Rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    return (Rs + Rp) / 2, theta_t


def get_mean_surface_position(logger, solidLabel, surfaceLabel, leaving=True):
    """Mean position of photons crossing a surface."""
    factory = PointCloudFactory(logger)
    pc = factory.getPointCloud(solidLabel, surfaceLabel)
    points = pc.leavingSurfacePoints if leaving else pc.enteringSurfacePoints
    if points is None or len(points) == 0:
        return None
    return np.mean(points[:, 1:4], axis=0)


def _get_surface_points_with_ids(logger, solidLabel, surfaceLabel, leaving=True):
    factory = PointCloudFactory(logger)
    pc = factory.getPointCloud(solidLabel, surfaceLabel)
    points = pc.leavingSurfacePoints if leaving else pc.enteringSurfacePoints
    if points is None or len(points) == 0 or points.shape[1] < 5:
        return None
    # Keep each photon's FIRST crossing to avoid re-crossings from Fresnel reflections.
    ids = points[:, 4].astype(np.int64)
    seen = {}
    for i, pid in enumerate(ids):
        if pid not in seen:
            seen[pid] = i
    idx = np.array(list(seen.values()), dtype=np.int64)
    return points[idx][:, [4, 1, 2, 3]]  # (id, x, y, z)


def per_photon_angle_median(logger, solidA, surfA, leavingA, solidB, surfB, leavingB):
    """Median beam angle (xz-plane) for photons crossing both surfaces.

    Direction = r_exit - r_entry per photon. In a non-scattering layer this is the exact
    ray direction, so the distribution collapses around the Snell angle with no method
    bias. The median absorbs the <~3% Fresnel-reflected tail.
    """
    a = _get_surface_points_with_ids(logger, solidA, surfA, leaving=leavingA)
    b = _get_surface_points_with_ids(logger, solidB, surfB, leaving=leavingB)
    if a is None or b is None:
        return None, 0
    common = np.intersect1d(a[:, 0], b[:, 0], assume_unique=True)
    if common.size == 0:
        return None, 0
    a_map = {int(r[0]): r[1:4] for r in a}
    b_map = {int(r[0]): r[1:4] for r in b}
    angles = []
    for pid in common:
        pid = int(pid)
        d = b_map[pid] - a_map[pid]
        dz, dx = d[2], d[0]
        if abs(dz) < 1e-12:
            continue
        angles.append(math.degrees(math.atan2(abs(dx), abs(dz))))
    if not angles:
        return None, 0
    return float(np.median(angles)), len(angles)


def measure_beam_angle_from_positions(pos1, pos2):
    """Beam angle from normal (z-axis) between two mean surface-crossing positions."""
    delta = pos2 - pos1
    return math.degrees(math.atan2(abs(delta[0]), abs(delta[2])))


def verify_all(logger, n_world, n_layers, mu_a, thicknesses, theta_i_deg):
    """Verify Snell angles, Fresnel reflections, and Beer-Lambert absorbance."""
    stats = Stats(logger)
    all_passed = True

    # --- Snell's law predictions at each interface ---
    theta = math.radians(theta_i_deg)
    interfaces = [
        ("Air -> Layer 1", n_world, n_layers[0]),
        ("Layer 1 -> Layer 2", n_layers[0], n_layers[1]),
        ("Layer 2 -> Layer 3", n_layers[1], n_layers[2]),
        ("Layer 3 -> Air", n_layers[2], n_world),
    ]
    snell_angles = []  # (theta_in, theta_out, R) at each interface
    for _, n1, n2 in interfaces:
        R, theta_t = fresnel_unpolarized(n1, n2, theta)
        snell_angles.append((theta, theta_t, R))
        theta = theta_t

    # --- Mean positions at each forward surface crossing ---
    surface_crossings = [
        ("frontLayer", "frontLayer_front", False),   # entering front
        ("frontLayer", "interface1", True),           # leaving front -> entering middle
        ("middleLayer", "interface0", True),          # leaving middle -> entering back
        ("backLayer", "backLayer_back", True),        # leaving back
    ]
    positions = [get_mean_surface_position(logger, s, surf, lv) for s, surf, lv in surface_crossings]

    print("\n" + "=" * 78)
    print("VERIFICATION: SNELL'S LAW (refraction angles)")
    print("=" * 78)
    print(f"  {'Segment':<22} {'Theory':>8} {'MeanPos':>8} {'Δmp':>7} "
          f"{'PerPhoton':>11} {'Δpp':>8} {'N_pp':>8} {'Status':>7}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*11} {'-'*8} {'-'*8} {'-'*7}")

    segment_labels = [
        "In Layer 1 (Air->L1)",
        "In Layer 2 (L1->L2)",
        "In Layer 3 (L2->L3)",
    ]
    pp_segments = [
        ("frontLayer",  "frontLayer_front", False, "frontLayer",  "interface1",     True),
        ("frontLayer",  "interface1",       True,  "middleLayer", "interface0",     True),
        ("middleLayer", "interface0",       True,  "backLayer",   "backLayer_back", True),
    ]
    for i in range(len(positions) - 1):
        if positions[i] is None or positions[i + 1] is None:
            print(f"  {segment_labels[i]:<22} {'N/A':>8}")
            continue
        theory_angle = math.degrees(snell_angles[i][1])
        mp_angle = measure_beam_angle_from_positions(positions[i], positions[i + 1])
        delta_mp = abs(mp_angle - theory_angle)

        pp_angle, n_pp = per_photon_angle_median(logger, *pp_segments[i])
        pp_str = f"{pp_angle:>10.3f}°" if pp_angle is not None else f"{'N/A':>11}"
        dpp_str = f"{abs(pp_angle - theory_angle):>7.3f}°" if pp_angle is not None else f"{'N/A':>8}"

        # Authoritative check = per-photon median (tight tol); fall back to mean-pos.
        if pp_angle is not None:
            passed = abs(pp_angle - theory_angle) < PER_PHOTON_ANGLE_TOL
        else:
            passed = delta_mp < MEANPOS_ANGLE_TOL
        all_passed &= passed
        print(f"  {segment_labels[i]:<22} {theory_angle:>7.2f}° {mp_angle:>7.2f}° "
              f"{delta_mp:>6.2f}° {pp_str} {dpp_str} {n_pp:>8d} {'PASS' if passed else 'FAIL':>7}")

    # --- Fresnel reflection + total transmission ---
    print("\n" + "=" * 78)
    print("VERIFICATION: FRESNEL REFLECTIONS")
    print("=" * 78)
    print(f"  {'Interface':<30} {'Theory':>10} {'Sim.':>10} {'Δ':>8} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    R_theory = snell_angles[0][2] * 100
    sim_refl = stats.getTransmittance("frontLayer", "frontLayer_front")
    delta = abs(sim_refl - R_theory)
    passed = delta < PERCENT_TOL
    all_passed &= passed
    print(f"  {'Air -> Layer 1 (reflection)':<30} {R_theory:>9.2f}% {sim_refl:>9.2f}% "
          f"{delta:>7.2f}% {'PASS' if passed else 'FAIL':>8}")

    cumulative_T = 1.0
    for i, _ in enumerate(interfaces):
        cumulative_T *= (1 - snell_angles[i][2])
        if i < len(thicknesses):
            path_length = thicknesses[i] / math.cos(snell_angles[i][1])
            cumulative_T *= math.exp(-mu_a[i] * path_length)
    sim_transmitted = stats.getTransmittance("backLayer", "backLayer_back", useTotalEnergy=True)
    delta = abs(sim_transmitted - cumulative_T * 100)
    passed = delta < PERCENT_TOL
    all_passed &= passed
    print(f"  {'Total transmission':<30} {cumulative_T * 100:>9.2f}% {sim_transmitted:>9.2f}% "
          f"{delta:>7.2f}% {'PASS' if passed else 'FAIL':>8}")

    # --- Beer-Lambert absorbance per layer ---
    print("\n" + "=" * 78)
    print("VERIFICATION: BEER-LAMBERT ABSORBANCE")
    print("=" * 78)
    print(f"  {'Layer':<30} {'Theory':>10} {'Sim.':>10} {'Δ':>8} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for i, label in enumerate(["frontLayer", "middleLayer", "backLayer"]):
        path_length = thicknesses[i] / math.cos(snell_angles[i][1])
        beer_abs = (1 - math.exp(-mu_a[i] * path_length)) * 100
        sim_abs = stats.getAbsorbance(label)
        delta = abs(sim_abs - beer_abs)
        passed = delta < PERCENT_TOL
        all_passed &= passed
        print(f"  {label:<30} {beer_abs:>9.2f}% {sim_abs:>9.2f}% {delta:>7.2f}% "
              f"{'PASS' if passed else 'FAIL':>8}")

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    print("=" * 78 + "\n")
    return all_passed


# ======================================================================================
#  Scene builders
# ======================================================================================
def _make_source(N):
    angle_rad = math.radians(ANGLE_DEG)
    direction = Vector(math.sin(angle_rad), 0, math.cos(angle_rad))
    return PencilPointSource(
        position=Vector(*SOURCE_POSITION), direction=direction, N=N, displaySize=SOURCE_DISPLAY_SIZE
    )


def build_refraction_scene(N):
    """Three-layer stack with mu_s = 0 (PhantomTissue geometry, scattering disabled)."""
    n = [1.4, 1.7, 1.4]
    mu_a = [0.1, 0.1, 0.1]
    g = 0
    w = 3
    t = [0.75, 0.5, 0.75]

    frontLayer = Cuboid(w, w, t[0], material=ScatteringMaterial(0, mu_a[0], g, n[0]), label="frontLayer")
    middleLayer = Cuboid(w, w, t[1], material=ScatteringMaterial(0, mu_a[1], g, n[1]), label="middleLayer")
    backLayer = Cuboid(w, w, t[2], material=ScatteringMaterial(0, mu_a[2], g, n[2]), label="backLayer")
    layerStack = backLayer.stack(middleLayer, "front").stack(frontLayer, "front")
    layerStack.translateTo(Vector(0, 0, sum(t) / 2))

    tissue = ScatteringScene([layerStack])
    logger = EnergyLogger(tissue)
    source = _make_source(N)
    source.propagate(tissue, logger=logger)
    return tissue, source, logger, dict(n=n, mu_a=mu_a, t=t, n_world=1.0)


def build_scattering_scene(N):
    """PhantomTissue: same geometry as above but with scattering enabled."""
    tissue = samples.PhantomTissue()
    logger = EnergyLogger(tissue)
    source = _make_source(N)
    source.propagate(tissue, logger=logger)
    return tissue, source, logger


# ======================================================================================
#  Matched 3D rendering
# ======================================================================================
def render_3d(scene, source, logger, out_path, figname, *, show, style=None):
    """Render one scene with the shared pinned camera / FOV / white bg, save and (opt) show.

    Reuses the same scene + source + point-cloud drawing as ``Viewer.show3D`` but pins the
    camera explicitly so the panel registers with the other one.
    """
    import mayavi.mlab as mlab

    style = style or PointCloudStyle(colormap=BEAM_COLORMAP, reverseColormap=BEAM_REVERSE)

    # Fresh figure (fixed size) so renders don't collide across panels.
    fig = mlab.figure(figname, size=RENDER_SIZE, bgcolor=BG_COLOR, fgcolor=FG_COLOR)

    viewer3D = get3DViewer()                       # constructor resets bg to dark...
    viewer3D.setViewPointStyle(ViewPointStyle.OPTICS)
    scene.addToViewer(viewer3D, opacity=SOLID_OPACITY, color=SOLID_COLOR)
    source.addToViewer(viewer3D, color=SOURCE_COLOR)

    # Same point-cloud rendering as viewer.show3D() (solid + surface clouds).
    v = Viewer(scene, source, logger)
    v._viewer3D = viewer3D
    v._addPointCloud(style)

    # ...so re-apply the white background AFTER populating.
    mlab.figure(fig, bgcolor=BG_COLOR, fgcolor=FG_COLOR)

    # Pin the identical camera + FOV for both panels.
    mlab.view(
        azimuth=CAMERA["azimuth"],
        elevation=CAMERA["elevation"],
        distance=CAMERA["distance"],
        focalpoint=CAMERA["focalpoint"],
        roll=CAMERA["roll"],
        figure=fig,
    )
    fig.scene.camera.parallel_projection = False
    fig.scene.camera.view_angle = VIEW_ANGLE       # identical field of view
    fig.scene.render()

    mlab.savefig(out_path, size=RENDER_SIZE, magnification=1)
    print(f"  saved {out_path}")

    if show:
        mlab.show()
    else:
        mlab.close(fig)
    return out_path


def combine_side_by_side(path_a, path_b, out_path):
    """Stitch the two registered panels into one (a)|(b) figure."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    ia, ib = mpimg.imread(path_a), mpimg.imread(path_b)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, im, title in zip(
        axes, [ia, ib], ["(a) Refraction  (μs = 0)", "(b) Scattering  (PhantomTissue)"]
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.02)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path}")


# ======================================================================================
#  Main
# ======================================================================================
def exampleCode(show=True, save=True):
    accel = hardwareAccelerationIsAvailable()
    N_REFRACTION = 1_000_000 if accel else 1000
    N_SCATTER = 100_000 if accel else 1000

    print("\n" + "#" * 78)
    print("# Pencil beam: refraction (μs = 0) vs scattering (PhantomTissue)")
    print(f"# Hardware acceleration: {accel} | N_refraction = {N_REFRACTION:,} | N_scatter = {N_SCATTER:,}")
    print("#" * 78)

    # --- (a) Refraction: propagate, report, verify ---
    print("\n[1/2] Refraction scene (μs = 0) ...")
    ref_scene, ref_source, ref_logger, p = build_refraction_scene(N_REFRACTION)
    Stats(ref_logger).report()
    verify_all(ref_logger, p["n_world"], p["n"], p["mu_a"], p["t"], ANGLE_DEG)

    # --- (b) Scattering: propagate, report ---
    print("[2/2] Scattering scene (PhantomTissue) ...")
    sca_scene, sca_source, sca_logger = build_scattering_scene(N_SCATTER)
    Stats(sca_logger).report()

    # --- Matched 3D panels ---
    print("\n" + "=" * 78)
    print("3D RENDER (shared camera / FOV / white background)")
    print("=" * 78)
    print(f"  camera     : azimuth={CAMERA['azimuth']}, elevation={CAMERA['elevation']}, "
          f"distance={CAMERA['distance']}, focalpoint={CAMERA['focalpoint']}, roll={CAMERA['roll']}")
    print(f"  view_angle : {VIEW_ANGLE}°   render_size: {RENDER_SIZE}   bgcolor: {BG_COLOR}")

    if not (show or save):
        return

    path_a = os.path.join(OUT_DIR, "fig1_combined_refraction.png")
    path_b = os.path.join(OUT_DIR, "fig1_combined_scattering.png")
    path_ab = os.path.join(OUT_DIR, "fig1_combined_side_by_side.png")
    try:
        render_3d(ref_scene, ref_source, ref_logger, path_a, "refraction", show=show)
        render_3d(sca_scene, sca_source, sca_logger, path_b, "scattering", show=show)
        if save:
            combine_side_by_side(path_a, path_b, path_ab)
    except Exception as e:
        print(f"  3D rendering skipped/failed ({type(e).__name__}: {e}). "
              f"Console verification above is unaffected.")


if __name__ == "__main__":
    exampleCode(show=True, save=True)
