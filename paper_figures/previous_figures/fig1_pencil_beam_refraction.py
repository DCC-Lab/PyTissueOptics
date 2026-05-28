"""
Figure 1 – Pencil beam refraction through a three-layer stack with μs = 0.

The beam path (color map: photon energy) follows Snell's law at each interface.
The reflected beam at the first surface is visible at lower left.
Layer boundaries and refractive indices are annotated.
"""

import math

import numpy as np

from pytissueoptics import *  # noqa: F403
from pytissueoptics.rayscattering.statistics import Stats
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

TITLE = "Pencil beam refraction through a three-layer stack (no scattering)"

DESCRIPTION = """ Propagation of a pencil beam through a three-layer stack with μs=0 (no scattering).
Same geometry as PhantomTissue (n=[1.4, 1.7, 1.4]) but with scattering disabled to isolate
Snell's law refraction at each interface. """


def fresnel_unpolarized(n1, n2, theta_i):
    """Fresnel reflection coefficient for unpolarized light."""
    theta_t = math.asin((n1 / n2) * math.sin(theta_i))
    cos_i, cos_t = math.cos(theta_i), math.cos(theta_t)
    Rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    Rp = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    return (Rs + Rp) / 2, theta_t


def get_mean_surface_position(logger, solidLabel, surfaceLabel, leaving=True):
    """Get the mean position of photons crossing a surface."""
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
    """Median beam angle in the xz-plane for photons that crossed both surfaces.

    Direction = r_exit - r_entry per photon.  In a non-scattering layer this is
    the exact ray direction, so the distribution collapses around the Snell
    angle with zero method bias.  Median absorbs the <~3% Fresnel-reflected tail.
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
    """Measure beam angle from normal (z-axis) between two surface crossing positions."""
    delta = pos2 - pos1
    dz = delta[2]
    dx = delta[0]
    return math.degrees(math.atan2(abs(dx), abs(dz)))


def verify_all(logger, n_world, n_layers, mu_a, thicknesses, theta_i_deg):
    """Verify Snell's law angles, Fresnel reflections, and Beer-Lambert absorbance."""
    stats = Stats(logger)
    all_passed = True
    ANGLE_TOL = 0.5  # degrees
    PERCENT_TOL = 1.0  # percentage points

    # --- Snell's law predictions ---
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

    # --- Measure beam angles from surface positions ---
    # Collect mean positions at each surface crossing (forward beam)
    surface_crossings = [
        ("frontLayer", "frontLayer_front", False),   # entering front
        ("frontLayer", "interface1", True),           # leaving front -> entering middle
        ("middleLayer", "interface0", True),          # leaving middle -> entering back
        ("backLayer", "backLayer_back", True),        # leaving back
    ]
    positions = []
    for solid, surface, leaving in surface_crossings:
        pos = get_mean_surface_position(logger, solid, surface, leaving)
        positions.append(pos)

    print("\n" + "=" * 70)
    print("VERIFICATION: SNELL'S LAW (refraction angles)")
    print("=" * 70)
    print(f"  {'Segment':<22} {'Theory':>8} {'MeanPos':>8} {'Δmp':>7} "
          f"{'PerPhoton':>10} {'Δpp':>7} {'N_pp':>7} {'Status':>7}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*7} {'-'*7} {'-'*7}")

    segment_labels = [
        "In Layer 1 (Air->L1)",
        "In Layer 2 (L1->L2)",
        "In Layer 3 (L2->L3)",
        "After exit (L3->Air)",
    ]

    pp_segments = [
        ("frontLayer",  "frontLayer_front", False, "frontLayer",  "interface1",       True),
        ("frontLayer",  "interface1",       True,  "middleLayer", "interface0",       True),
        ("middleLayer", "interface0",       True,  "backLayer",   "backLayer_back",   True),
    ]
    PER_PHOTON_TOL = 0.02
    for i in range(len(positions) - 1):
        if positions[i] is None or positions[i + 1] is None:
            print(f"  {segment_labels[i]:<22} {'N/A':>8}")
            continue
        theory_angle = math.degrees(snell_angles[i][1])
        mp_angle = measure_beam_angle_from_positions(positions[i], positions[i + 1])
        delta_mp = abs(mp_angle - theory_angle)

        pp_angle, n_pp = (None, 0)
        if i < len(pp_segments):
            a1, a2, a3, b1, b2, b3 = pp_segments[i]
            pp_angle, n_pp = per_photon_angle_median(logger, a1, a2, a3, b1, b2, b3)
        pp_str = f"{pp_angle:>9.3f}°" if pp_angle is not None else f"{'N/A':>10}"
        dpp_str = f"{abs(pp_angle - theory_angle):>6.3f}°" if pp_angle is not None else f"{'N/A':>7}"

        reference_delta = abs(pp_angle - theory_angle) if pp_angle is not None else delta_mp
        passed = reference_delta < PER_PHOTON_TOL if pp_angle is not None else delta_mp < ANGLE_TOL
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {segment_labels[i]:<22} {theory_angle:>7.2f}° {mp_angle:>7.2f}° "
              f"{delta_mp:>6.2f}° {pp_str} {dpp_str} {n_pp:>7d} {status:>7}")

    # --- Fresnel reflection verification ---
    print("\n" + "=" * 70)
    print("VERIFICATION: FRESNEL REFLECTIONS")
    print("=" * 70)
    print(f"  {'Interface':<30} {'Theory':>10} {'Sim.':>10} {'Δ':>8} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    # First surface reflection = frontLayer_front transmittance
    R_theory = snell_angles[0][2] * 100
    sim_refl = stats.getTransmittance("frontLayer", "frontLayer_front")
    delta = abs(sim_refl - R_theory)
    passed = delta < PERCENT_TOL
    if not passed:
        all_passed = False
    print(f"  {'Air -> Layer 1 (reflection)':<30} {R_theory:>9.2f}% {sim_refl:>9.2f}% {delta:>7.2f}% {'PASS' if passed else 'FAIL':>8}")

    # Overall transmission
    cumulative_T = 1.0
    for i, (_, n1, n2) in enumerate(interfaces):
        cumulative_T *= (1 - snell_angles[i][2])
        if i < len(thicknesses):
            path_length = thicknesses[i] / math.cos(snell_angles[i][1])
            cumulative_T *= math.exp(-mu_a[i] * path_length)

    sim_transmitted = stats.getTransmittance("backLayer", "backLayer_back", useTotalEnergy=True)
    delta = abs(sim_transmitted - cumulative_T * 100)
    passed = delta < PERCENT_TOL
    if not passed:
        all_passed = False
    print(f"  {'Total transmission':<30} {cumulative_T * 100:>9.2f}% {sim_transmitted:>9.2f}% {delta:>7.2f}% {'PASS' if passed else 'FAIL':>8}")

    # --- Beer-Lambert absorbance verification ---
    print("\n" + "=" * 70)
    print("VERIFICATION: BEER-LAMBERT ABSORBANCE")
    print("=" * 70)
    print(f"  {'Layer':<30} {'Theory':>10} {'Sim.':>10} {'Δ':>8} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    layer_labels = ["frontLayer", "middleLayer", "backLayer"]
    for i, label in enumerate(layer_labels):
        theta_in_layer = snell_angles[i][1]
        path_length = thicknesses[i] / math.cos(theta_in_layer)
        beer_abs = (1 - math.exp(-mu_a[i] * path_length)) * 100
        sim_abs = stats.getAbsorbance(label)
        delta = abs(sim_abs - beer_abs)
        passed = delta < PERCENT_TOL
        if not passed:
            all_passed = False
        print(f"  {label:<30} {beer_abs:>9.2f}% {sim_abs:>9.2f}% {delta:>7.2f}% {'PASS' if passed else 'FAIL':>8}")

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 70 + "\n")

    return all_passed


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 1000

    # Three-layer stack matching PhantomTissue geometry but with μs=0
    n = [1.4, 1.7, 1.4]
    mu_a = [0.1, 0.1, 0.1]
    g = 0
    n_world = 1.0

    w = 3
    t = [0.75, 0.5, 0.75]

    frontLayer = Cuboid(w, w, t[0], material=ScatteringMaterial(0, mu_a[0], g, n[0]), label="frontLayer")
    middleLayer = Cuboid(w, w, t[1], material=ScatteringMaterial(0, mu_a[1], g, n[1]), label="middleLayer")
    backLayer = Cuboid(w, w, t[2], material=ScatteringMaterial(0, mu_a[2], g, n[2]), label="backLayer")
    layerStack = backLayer.stack(middleLayer, "front").stack(frontLayer, "front")
    layerStack.translateTo(Vector(0, 0, sum(t) / 2))

    tissue = ScatteringScene([layerStack])
    logger = EnergyLogger(tissue)

    angle_deg = 30
    angle_rad = math.radians(angle_deg)
    direction = Vector(math.sin(angle_rad), 0, math.cos(angle_rad))

    source = PencilPointSource(
        position=Vector(-0.5, 0, -0.3), direction=direction, N=N, displaySize=0.3
    )

    tissue.show(source=source)

    source.propagate(tissue, logger=logger)

    viewer = Viewer(tissue, source, logger)
    viewer.reportStats()

    # Automated verification (must run before show3D which modifies point cloud data)
    verify_all(logger, n_world, n, mu_a, t, angle_deg)

    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionX(energyType=EnergyType.FLUENCE_RATE))
    viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
    viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface1", surfaceEnergyLeaving=False))
    viewer.show1D(Direction.Z_POS)
    viewer.show3D()
    viewer.show3D(pointCloudStyle=PointCloudStyle(showSolidPoints=False))


if __name__ == "__main__":
    exampleCode()
