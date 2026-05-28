"""
Figure 4b - Sampling-volume (CW Jacobian / PMDF) on Colin27 with fNIRS geometry.

Computes the CONTINUOUS-WAVE JACOBIAN (equivalently, photon measurement density
function / partial pathlength) for three source-detector pairs on the Colin27
atlas at separations 20, 30, 40 mm.

Formal definition (Arridge 1995, Boas & Dale 2005):

    J_mu_a(Omega) = dOD / dmu_a(Omega) = <L(Omega)>_detected
                  = (1 / N_det) * sum_{k in det} w_k * L_k(Omega)

where L_k(Omega) is the ray-segment length of detected photon k inside mesh
element Omega, and w_k is the photon weight at detection. Units: mm of mean
partial pathlength per detected photon per element.

Computation in PyTissueOptics. Per scatter event the logger stores
delta = W * mu_a/mu_t at the event position. Dividing by the local mu_a and
summing inside a bin gives sum (W/mu_t) per bin, which approximates the
weight-weighted pathlength sum(W * Delta_s) in the scattering-dominated regime.
See energyLogger._fluenceTransform and EnergyType.FLUENCE_RATE.

    J(Omega) ~= (1 / N_det) * sum_{events in Omega} (delta_event / mu_a_local)

Known bias. PyTissueOptics logs only SCATTER endpoints, not continuous ray
sub-segments per element. MMC (Fang 2010) accumulates analytically along each
Plucker segment crossing, which is robust in low-scattering tissue. Our
point-binning under-samples the CSF layer (mu_s = 0.01 mm-1, MFP ~70 mm):
photons can stream through CSF with 0 or 1 scatter events, so partial
pathlength in CSF is under-estimated. Scalp, gray and white matter
(MFP < 0.1-0.12 mm) are accurate.

Presentation follows Strangman et al. PLoS ONE 2013, Brigadoi & Cooper
Neurophotonics 2015: sagittal mid-plane slice of log10(J/J_peak) with
5-decade dynamic range and decade contours (10^-1 ... 10^-5); per-layer
fractional sensitivity (%) bar chart; per-layer partial pathlength (mm)
table; depth-resolved partial pathlength profile.

Setup. Three Circle detectors (radius 3 mm) embedded 1 mm beneath the scalp
surface, set asDetector(halfAngle = pi/2). All three active in one run: photons
absorbed at the 20 mm detector cannot reach 30 or 40 mm. For a sub-mm detector
this bias is <0.1%. For publication numbers, rerun with one detector per
simulation.

Usage.
    python fig4b_sampling_volume_comparison.py                  # all figures
    python fig4b_sampling_volume_comparison.py 1 3              # subset
    SAMPLING_VOL_SAVE_DIR=/tmp/out python fig4b_...             # save PNGs
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from pytissueoptics import (
    DivergentSource,
    EnergyLogger,
    ScatteringMaterial,
    ScatteringScene,
    Vector,
    hardwareAccelerationIsAvailable,
)
from pytissueoptics.rayscattering.energyLogging import EnergyType
from pytissueoptics.scene.geometry import Environment, SurfaceCollection, Triangle, Vertex
from pytissueoptics.scene.logger import InteractionKey
from pytissueoptics.scene.solids import Circle, Solid

try:
    import scipy.io
except ImportError:
    raise ImportError("scipy required: pip install scipy")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MESH_PATH = "/Users/williamboucher/Documents/pytissuemodelsDEC2025/PyTissueOptics/pytissueoptics/examples/benchmarks/MMC_Collins_Atlas_Mesh_Version_2L.mat"

WAVELENGTH = 830  # nm
OPTICAL_PROPERTIES = {
    "world": ScatteringMaterial(mu_s=0, mu_a=0, g=1.0, n=1.0),
    "scalp": ScatteringMaterial(mu_s=8.5, mu_a=0.016, g=0.91, n=1.45),
    "csf":   ScatteringMaterial(mu_s=0.01, mu_a=0.004, g=0.90, n=1.33),
    "gray":  ScatteringMaterial(mu_s=10.5, mu_a=0.032, g=0.90, n=1.40),
    "white": ScatteringMaterial(mu_s=41.0, mu_a=0.012, g=0.85, n=1.40),
}

# Tissue solid-label -> OPTICAL_PROPERTIES key
TISSUE_SOLID_TO_MAT = {
    "brain_scalp":       "scalp",
    "brain_csf":         "csf",
    "brain_grayMatter":  "gray",
    "brain_whiteMatter": "white",
}
TISSUE_DISPLAY = ["scalp", "CSF", "gray", "white"]

DETECTOR_SEPARATIONS_MM = [20, 30, 40]
DETECTOR_LABELS = [f"detector_{d}mm" for d in DETECTOR_SEPARATIONS_MM]

N_PHOTONS_GPU = 500_000
N_PHOTONS_CPU = 20_000


# --------------------------------------------------------------------------- #
# Scene setup (identical to fig4, with detectors as Circle + asDetector())
# --------------------------------------------------------------------------- #

def compute_vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    for face in faces:
        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        for idx in face:
            normals[idx] += normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return normals / norms


def load_colin27():
    data = scipy.io.loadmat(MESH_PATH)
    nodes = data["node"]
    faces = data["face"].copy()
    faces[:, :3] -= 1
    faces[:, 3] -= 1

    layer_config = [
        ("scalp",       0, "scalp", "world"),
        ("csf",         1, "csf",   "scalp"),
        ("grayMatter",  2, "gray",  "csf"),
        ("whiteMatter", 3, "white", "gray"),
    ]
    solids: List[Solid] = []
    scalp_vertices = scalp_normals = None

    for label, surface_id, inside_mat, outside_mat in layer_config:
        mask = faces[:, 3] == surface_id
        surface_faces = faces[mask][:, :3]
        vertex_indices = np.unique(surface_faces.flatten())
        global_to_local = {g: l for l, g in enumerate(vertex_indices)}
        local_faces = np.array([[global_to_local[v] for v in f] for f in surface_faces])
        local_vertices = nodes[vertex_indices]

        if surface_id == 0:
            scalp_vertices = local_vertices.copy()
            scalp_normals = compute_vertex_normals(local_vertices, local_faces)

        verts = [Vertex(*v) for v in local_vertices]
        tris = [Triangle(verts[f[2]], verts[f[1]], verts[f[0]]) for f in local_faces]
        surfaces = SurfaceCollection()
        surfaces.add(label, tris)

        solid = Solid(
            vertices=verts,
            surfaces=surfaces,
            label=f"brain_{label}",
            material=OPTICAL_PROPERTIES[inside_mat],
        )
        if surface_id == 0:
            solid.setOutsideEnvironment(Environment(material=OPTICAL_PROPERTIES[outside_mat]))
        else:
            solid.setOutsideEnvironment(
                Environment(material=OPTICAL_PROPERTIES[outside_mat], solid=solids[-1])
            )
        solids.append(solid)

    scene = ScatteringScene(solids=solids, ignoreIntersections=True)
    scalp_solid = solids[0]
    return scene, scalp_vertices, scalp_normals, scalp_solid


def find_scalp_point(scalp_vertices, scalp_normals, x, y, z_min=100):
    mask = scalp_vertices[:, 2] > z_min
    verts = scalp_vertices[mask]
    norms = scalp_normals[mask]
    dists = np.sqrt((verts[:, 0] - x) ** 2 + (verts[:, 1] - y) ** 2)
    idx = dists.argmin()
    return tuple(verts[idx]), tuple(norms[idx])


def make_detector(position, inward_normal, radius=3.0, embed_depth=1.0, label="detector"):
    # Disk embedded 1 mm inside scalp so both sides are in scalp material, which
    # keeps the OpenCL environment table consistent under ignoreIntersections.
    offset = Vector(
        position[0] + inward_normal[0] * embed_depth,
        position[1] + inward_normal[1] * embed_depth,
        position[2] + inward_normal[2] * embed_depth,
    )
    orient = Vector(*inward_normal)
    detector = Circle(radius=radius, orientation=orient, position=offset, label=label)
    detector.asDetector(halfAngle=np.pi / 2)
    return detector


# --------------------------------------------------------------------------- #
# Simulation wrapper
# --------------------------------------------------------------------------- #

@dataclass
class SimResult:
    scene: ScatteringScene
    source: DivergentSource
    logger: EnergyLogger
    source_pos: np.ndarray
    source_dir: np.ndarray              # inward unit vector
    source_y: float
    detector_positions: Dict[str, np.ndarray]
    n_launched: int


def simulate() -> SimResult:
    N = N_PHOTONS_GPU if hardwareAccelerationIsAvailable() else N_PHOTONS_CPU
    print(f"\n[setup] Loading Colin27 mesh ...")
    scene, scalp_vertices, scalp_normals, scalp_solid = load_colin27()

    src_x, src_y = 90, 105
    scalp_pos, scalp_normal = find_scalp_point(scalp_vertices, scalp_normals, src_x, src_y)

    source_pos_vec = Vector(
        scalp_pos[0] - scalp_normal[0] * 2,
        scalp_pos[1] - scalp_normal[1] * 2,
        scalp_pos[2] - scalp_normal[2] * 2,
    )
    source_dir_vec = Vector(*scalp_normal)

    source = DivergentSource(
        position=source_pos_vec,
        direction=source_dir_vec,
        diameter=0.8,
        divergence=0.22,
        N=N,
        displaySize=5,
    )

    detector_positions: Dict[str, np.ndarray] = {}
    for sep, label in zip(DETECTOR_SEPARATIONS_MM, DETECTOR_LABELS):
        det_pos, det_normal = find_scalp_point(
            scalp_vertices, scalp_normals, src_x + sep, src_y
        )
        detector = make_detector(det_pos, det_normal, label=label)
        detector.setOutsideEnvironment(scalp_solid.getEnvironment())
        scene.add(detector)
        detector_positions[label] = np.array(det_pos)
        print(f"[setup] {label} at ({det_pos[0]:.1f}, {det_pos[1]:.1f}, {det_pos[2]:.1f}) mm")

    print(f"[run] Propagating {N:,} photons ...")
    logger = EnergyLogger(scene)
    source.propagate(scene, logger)

    return SimResult(
        scene=scene,
        source=source,
        logger=logger,
        source_pos=np.array(scalp_pos),
        source_dir=np.array(scalp_normal),
        source_y=float(src_y),
        detector_positions=detector_positions,
        n_launched=N,
    )


# --------------------------------------------------------------------------- #
# Core: pathlength contribution per logged event
# --------------------------------------------------------------------------- #
# Per scatter event, energyLogger._fluenceTransform divides the logged delta
# energy by mu_a of the containing solid, giving delta/mu_a = W/mu_t = W*<step>.
# This is the weight-weighted mean-free-path contributed by that event. We
# concatenate these contributions across all tissue solids for photons detected
# by a given detector.

def _tissue_labels(logger: EnergyLogger) -> List[str]:
    return [s for s in logger.getStoredSolidLabels() if s in TISSUE_SOLID_TO_MAT]


def pathlength_events(logger: EnergyLogger) -> Dict[str, np.ndarray]:
    """Return, per tissue solid, an array (contribution, x, y, z) where
    contribution = delta_event / mu_a_local has units of millimetres (step
    length times photon weight). Concatenating all tissues gives the full set
    of pathlength contributions for the filtered photon population."""
    out: Dict[str, np.ndarray] = {}
    for tlabel in _tissue_labels(logger):
        pts = logger.getDataPoints(InteractionKey(tlabel), energyType=EnergyType.FLUENCE_RATE)
        if pts is None or len(pts) == 0:
            continue
        # getDataPoints may mutate; take a defensive copy of the 4 columns we need
        out[tlabel] = np.asarray(pts[:, :4], dtype=np.float64).copy()
    return out


def n_detected(sim: SimResult, detector_label: str) -> int:
    det_pts = sim.logger.getRawDataPoints(InteractionKey(detector_label))
    if det_pts is None or det_pts.shape[1] < 5:
        return 0
    return int(np.unique(det_pts[:, 4].astype(np.uint32)).size)


def filtered(sim: SimResult, detector_label: str) -> EnergyLogger:
    return sim.logger.getFiltered(detector_label)


# --------------------------------------------------------------------------- #
# Partial pathlength per tissue (mm per detected photon)
# --------------------------------------------------------------------------- #

def per_tissue_pathlength(sim: SimResult, detector_label: str) -> Dict[str, float]:
    """Mean partial pathlength <L_layer> in mm per detected photon, per tissue.
    This is the Boas-Dale partial pathlength that enters the modified
    Beer-Lambert law and determines the per-layer sensitivity."""
    n_det = n_detected(sim, detector_label)
    if n_det == 0:
        return {k: 0.0 for k in TISSUE_SOLID_TO_MAT.values()}
    flog = filtered(sim, detector_label)
    per_event = pathlength_events(flog)
    out = {}
    for tlabel, mat_key in TISSUE_SOLID_TO_MAT.items():
        arr = per_event.get(tlabel)
        if arr is None or len(arr) == 0:
            out[mat_key] = 0.0
            continue
        out[mat_key] = float(arr[:, 0].sum()) / n_det
    return out


# --------------------------------------------------------------------------- #
# 3D pathlength density rasterisation (post-processing, for visualisation only)
# --------------------------------------------------------------------------- #

def rasterise_jacobian(sim: SimResult, detector_label: str, bin_mm: float,
                       limits: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    """Accumulate weight-weighted pathlength contributions on a regular
    Cartesian grid. This is a DISPLAY step - the simulation remains mesh-based.

    Returns J[ix, iy, iz] = <L(bin)> in mm per detected photon per bin.
    Contributions are scatter-endpoint bins (see module docstring caveat)."""
    n_det = n_detected(sim, detector_label)
    flog = filtered(sim, detector_label)
    per_event = pathlength_events(flog)

    bins = [np.arange(lo, hi + bin_mm, bin_mm) for (lo, hi) in limits]
    shape = tuple(len(b) - 1 for b in bins)
    hist = np.zeros(shape, dtype=np.float64)

    for arr in per_event.values():
        if len(arr) == 0:
            continue
        h, _ = np.histogramdd(arr[:, 1:4], bins=bins, weights=arr[:, 0])
        hist += h
    if n_det > 0:
        hist /= n_det
    return hist, bins


def rasterise_deposition(sim: SimResult, bin_mm: float,
                          limits: Tuple[Tuple[float, float], ...]):
    """Same rasterisation as `rasterise_jacobian`, but *without* filtering by
    a detector — i.e. contributions from ALL launched photons.  Normalised by
    sim.n_launched so the return has the same units (mm of pathlength per
    photon per voxel) as the Jacobian, making the two directly comparable.

    This is the "illumination fluence" map: where light goes in aggregate,
    irrespective of whether it reached any detector.
    """
    per_event = pathlength_events(sim.logger)   # no detector filter

    bins = [np.arange(lo, hi + bin_mm, bin_mm) for (lo, hi) in limits]
    shape = tuple(len(b) - 1 for b in bins)
    hist = np.zeros(shape, dtype=np.float64)

    for arr in per_event.values():
        if len(arr) == 0:
            continue
        h, _ = np.histogramdd(arr[:, 1:4], bins=bins, weights=arr[:, 0])
        hist += h
    if sim.n_launched > 0:
        hist /= sim.n_launched
    return hist, bins


# --------------------------------------------------------------------------- #
# Figure 1 - sagittal slice of log10(J / J_peak) with decade contours
# --------------------------------------------------------------------------- #

def figure1_sagittal_jacobian(sim: SimResult, bin_mm: float = 0.5,
                               slab_half_thickness_mm: float = 2.0):
    """Sagittal mid-plane slice through source-detector axis (y = y_source).
    log10 of J normalised to peak, 5-decade dynamic range, contours at 10^-1
    down to 10^-5. One panel per detector. Strangman 2013 convention."""
    print("\n[fig 1] Sagittal Jacobian slice (log10 J/J_peak, decade contours) ...")

    xyz_limits = sim.scene.getBoundingBox().xyzLimits
    limits = (tuple(xyz_limits[0]), tuple(xyz_limits[1]), tuple(xyz_limits[2]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    contour_levels = [-5, -4, -3, -2, -1]  # decades below peak

    last_im = None
    for ax, det_label in zip(axes, DETECTOR_LABELS):
        hist, bins = rasterise_jacobian(sim, det_label, bin_mm, limits)
        # y-slab centred at source_y: marginalise over |y - y_src| <= slab_half
        yedges = bins[1]
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        ymask = np.abs(ycenters - sim.source_y) <= slab_half_thickness_mm
        if not ymask.any():
            # widen to nearest bin
            ymask = np.zeros_like(ycenters, dtype=bool)
            ymask[np.argmin(np.abs(ycenters - sim.source_y))] = True
        # slab sum: pathlength contributions across the y-slab thickness
        slab = hist[:, ymask, :].sum(axis=1)   # shape (nx, nz)
        peak = slab.max()
        if peak <= 0:
            ax.set_title(f"{det_label.split('_')[1]}: no detections")
            continue
        norm = slab / peak
        log_norm = np.log10(np.maximum(norm, 10 ** (contour_levels[0] - 1))).T  # (nz, nx)

        xedges, zedges = bins[0], bins[2]
        im = ax.imshow(
            log_norm,
            origin="lower",
            extent=(xedges[0], xedges[-1], zedges[0], zedges[-1]),
            aspect="equal", cmap="inferno",
            vmin=contour_levels[0], vmax=0,
        )
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        zcenters = 0.5 * (zedges[:-1] + zedges[1:])
        cs = ax.contour(
            xcenters, zcenters, log_norm,
            levels=contour_levels, colors="cyan", linewidths=0.8, alpha=0.7,
        )
        ax.clabel(cs, fmt=lambda v: f"10^{int(v)}", fontsize=7)
        ax.plot(sim.source_pos[0], sim.source_pos[2], "w*", ms=14, mec="k", mew=1.0)
        d_xy = sim.detector_positions[det_label]
        ax.plot(d_xy[0], d_xy[2], "wo", ms=9, mec="k", mew=1.0)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        sep = det_label.split("_")[1]
        ppl = sum(per_tissue_pathlength(sim, det_label).values())
        ax.set_title(f"{sep}  |  n_det = {n_detected(sim, det_label)}  |  <L> = {ppl:.1f} mm")
        last_im = im

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.8, label="log10(J / J_peak)")
    fig.suptitle(
        "CW Jacobian (sagittal slice, y = y_source +/- 2 mm): "
        "sampling volume of detected photons"
    )


# --------------------------------------------------------------------------- #
# Figure 2 - depth-resolved partial pathlength profile
# --------------------------------------------------------------------------- #

def figure2_depth_profile(sim: SimResult, bin_mm: float = 0.5):
    """Mean partial pathlength per detected photon vs depth from the scalp
    entry point along the inward normal. Units: mm. One curve per detector.
    Shows the fNIRS 'depth sensitivity' — where does each channel sample?"""
    print("\n[fig 2] Depth-resolved partial pathlength profile ...")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    inward = sim.source_dir / np.linalg.norm(sim.source_dir)
    depth_edges = np.arange(0.0, 80.0 + bin_mm, bin_mm)
    centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])

    for det_label in DETECTOR_LABELS:
        n_det = n_detected(sim, det_label)
        if n_det == 0:
            continue
        flog = filtered(sim, det_label)
        per_event = pathlength_events(flog)
        all_pts = np.concatenate([a for a in per_event.values()], axis=0) if per_event else np.empty((0, 4))
        if len(all_pts) == 0:
            continue
        rel = all_pts[:, 1:4] - sim.source_pos[None, :]
        depths = rel @ inward
        hist, _ = np.histogram(depths, bins=depth_edges, weights=all_pts[:, 0])
        hist /= n_det  # mean partial pathlength per detected photon per depth bin
        sep = det_label.split("_")[1]
        ax.plot(centers, hist, label=f"{sep}  (n_det={n_det})", lw=2)

    ax.set_xlabel("depth from scalp entry (mm)")
    ax.set_ylabel("mean partial pathlength per detected photon (mm / bin)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Depth sensitivity: per-detector partial pathlength vs depth")


# --------------------------------------------------------------------------- #
# Figure 3 - per-tissue partial pathlength (bar) + table
# --------------------------------------------------------------------------- #

def figure3_per_tissue(sim: SimResult):
    """Per-layer fractional sensitivity (percent bar chart) and partial
    pathlength table in mm. This is the literature-standard way to report
    depth sensitivity in fNIRS (Strangman 2013; Brigadoi 2015)."""
    print("\n[fig 3] Per-tissue partial pathlength (bar + table) ...")
    per_det_ppl = {d: per_tissue_pathlength(sim, d) for d in DETECTOR_LABELS}

    # Text table
    print()
    print(f"  {'sep':<6} {'n_det':>6} {'scalp (mm)':>11} {'CSF (mm)':>9} "
          f"{'gray (mm)':>10} {'white (mm)':>11} {'total L (mm)':>13} "
          f"{'DPF':>6}")
    print("  " + "-" * 80)
    for det_label in DETECTOR_LABELS:
        ppl = per_det_ppl[det_label]
        sep = int(det_label.split("_")[1].rstrip("mm"))
        n_det = n_detected(sim, det_label)
        total_L = sum(ppl.values())
        dpf = total_L / sep if sep else 0.0
        print(
            f"  {sep:<6} {n_det:>6} {ppl['scalp']:>11.3f} {ppl['csf']:>9.3f} "
            f"{ppl['gray']:>10.3f} {ppl['white']:>11.3f} {total_L:>13.3f} "
            f"{dpf:>6.2f}"
        )

    # Bar chart: fractional sensitivity (%)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = np.arange(len(DETECTOR_LABELS))
    bottom = np.zeros(len(DETECTOR_LABELS))
    colors = {"scalp": "#f0a070", "csf": "#7ec8e3", "gray": "#b0b0b0", "white": "#f5f5f5"}
    for mat_key, disp in zip(["scalp", "csf", "gray", "white"], TISSUE_DISPLAY):
        vals = np.array([per_det_ppl[d][mat_key] for d in DETECTOR_LABELS])
        totals = np.array([sum(per_det_ppl[d].values()) for d in DETECTOR_LABELS])
        frac = np.where(totals > 0, 100 * vals / totals, 0)
        ax.bar(x, frac, bottom=bottom, label=disp, color=colors[mat_key], edgecolor="k")
        bottom += frac
    ax.set_xticks(x)
    ax.set_xticklabels([d.split("_")[1] for d in DETECTOR_LABELS])
    ax.set_ylabel("fractional sensitivity per layer (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-tissue fractional sensitivity (detected-photon pathlength fraction)")
    ax.legend(title="tissue", bbox_to_anchor=(1.02, 1.0), loc="upper left")


# --------------------------------------------------------------------------- #
# Figure 4 - summary table printed to stdout
# --------------------------------------------------------------------------- #

def figure4_summary_table(sim: SimResult, bin_mm: float = 1.0):
    """Printed summary: n_det, total partial pathlength (mm), DPF, per-layer
    partial pathlength (mm), depth at 50% cumulative sensitivity, and the
    bin-sum volume enclosing 90% of Jacobian mass."""
    print("\n[fig 4] Summary table (mean partial pathlength per detected photon) ...")

    xyz_limits = sim.scene.getBoundingBox().xyzLimits
    limits = (tuple(xyz_limits[0]), tuple(xyz_limits[1]), tuple(xyz_limits[2]))

    inward = sim.source_dir / np.linalg.norm(sim.source_dir)

    print()
    print(f"  {'sep':<4} {'n_det':>6} {'<L>_tot':>9} {'DPF':>5} "
          f"{'<L>_scalp':>10} {'<L>_CSF':>8} {'<L>_gray':>9} {'<L>_white':>10} "
          f"{'z50 (mm)':>9} {'V90 (mm3)':>10}")
    print("  " + "-" * 95)

    for det_label in DETECTOR_LABELS:
        sep = int(det_label.split("_")[1].rstrip("mm"))
        n_det = n_detected(sim, det_label)
        ppl = per_tissue_pathlength(sim, det_label)
        tot = sum(ppl.values())
        dpf = tot / sep if sep else 0.0

        # Depth of 50% cumulative sensitivity
        flog = filtered(sim, det_label)
        per_event = pathlength_events(flog)
        if per_event and n_det > 0:
            allpts = np.concatenate(list(per_event.values()), axis=0)
            depths = (allpts[:, 1:4] - sim.source_pos[None, :]) @ inward
            w = allpts[:, 0]
            order = np.argsort(depths)
            cum = np.cumsum(w[order]) / w.sum()
            z50 = float(depths[order][np.searchsorted(cum, 0.5)])
        else:
            z50 = 0.0

        # V90 from rasterised Jacobian
        hist, bins = rasterise_jacobian(sim, det_label, bin_mm, limits)
        flat = np.sort(hist.ravel())[::-1]
        if flat.sum() > 0:
            cum = np.cumsum(flat) / flat.sum()
            n_bins = int(np.searchsorted(cum, 0.90) + 1)
            v90 = n_bins * (bin_mm ** 3)
        else:
            v90 = 0.0

        print(
            f"  {sep:<4} {n_det:>6} {tot:>8.2f} mm {dpf:>4.2f}  "
            f"{ppl['scalp']:>9.3f} {ppl['csf']:>7.3f} {ppl['gray']:>8.3f} "
            f"{ppl['white']:>9.3f}  {z50:>8.2f}  {v90:>9.1f}"
        )

    print(
        "\n  Columns:\n"
        "    n_det       - number of detected photons\n"
        "    <L>_tot     - total mean partial pathlength per detected photon (mm)\n"
        "    DPF         - differential pathlength factor = <L>_tot / sep\n"
        "    <L>_layer   - partial pathlength per layer per detected photon (mm)\n"
        "    z50         - depth (from scalp entry, along inward normal) enclosing 50% of sensitivity\n"
        "    V90         - volume (mm^3) of bins enclosing 90% of total Jacobian mass\n"
        "\n  Caveat: partial pathlengths are computed from scatter-event endpoints.\n"
        "  CSF is under-estimated because mu_s=0.01 mm^-1 gives a 70 mm mean free path;\n"
        "  MMC's continuous per-tet accumulation avoids this bias but is not available here.\n"
    )


# --------------------------------------------------------------------------- #
# Figure 6 - side-by-side: illumination vs detector-filtered sampling volume
# --------------------------------------------------------------------------- #
# Addresses a reviewer's request to show the distinction between
#   (a) the energy deposited in the tissue BY ALL LAUNCHED PHOTONS — a
#       broad diffuse cloud that surrounds the scalp entry point, and
#   (b) the energy deposited BY ONLY THOSE PHOTONS THAT REACHED THE DETECTOR
#       — the banana-shaped sampling volume between source and detector.
# Same slab, same colormap, same 5-decade log range on both panels so the
# drastic spatial constraint of the detected-photon map is immediately
# visible.  Left tile shows what the tissue *sees*; right tile shows what
# the measurement *reports*.

def figure6_deposition_vs_jacobian(sim: SimResult, bin_mm: float = 1.0,
                                    slab_half_thickness_mm: float = 3.0,
                                    detector_label: str = "detector_20mm",
                                    smooth_sigma_mm: float = 1.5):
    """Publication-quality side-by-side of illumination (all photons) vs
    detected-photon sampling volume.

    Design choices:
    * 1 mm voxels (coarser than 0.5 mm) so each voxel aggregates more
      photons — reduces shot noise in the right panel.
    * 3 mm slab (thicker than 2 mm) for the same reason.
    * Gaussian smoothing (σ ≈ 1.5 mm ≈ 1 voxel width in log-intensity space)
      removes single-voxel flicker without distorting the overall shape.
    * Matching rendering between panels: identical colormap ('magma'),
      identical log10 range [-4, 0], identical contour levels.  The two
      panels use filled contours rather than imshow — smoother, no hard
      voxel edges.
    * Scalp surface of the Colin27 mesh overlaid as a thin gray outline so
      the reader sees the anatomy.
    * Clean typography: short titles, single horizontal colorbar below both
      panels, no in-plot peak annotations (values go in caption).
    """
    from scipy.ndimage import gaussian_filter
    print("\n[fig 6] Deposition (all photons) vs Jacobian (detected) side-by-side ...")

    xyz_limits = sim.scene.getBoundingBox().xyzLimits
    limits = (tuple(xyz_limits[0]), tuple(xyz_limits[1]), tuple(xyz_limits[2]))

    hist_dep, bins = rasterise_deposition(sim, bin_mm, limits)
    hist_jac, _ = rasterise_jacobian(sim, detector_label, bin_mm, limits)

    yedges = bins[1]
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    ymask = np.abs(ycenters - sim.source_y) <= slab_half_thickness_mm
    if not ymask.any():
        ymask = np.zeros_like(ycenters, dtype=bool)
        ymask[np.argmin(np.abs(ycenters - sim.source_y))] = True

    slab_dep = hist_dep[:, ymask, :].sum(axis=1)
    slab_jac = hist_jac[:, ymask, :].sum(axis=1)

    # Gaussian smoothing to remove voxel-level shot noise before display.
    sigma_pix = smooth_sigma_mm / bin_mm
    slab_dep_s = gaussian_filter(slab_dep, sigma=sigma_pix)
    slab_jac_s = gaussian_filter(slab_jac, sigma=sigma_pix)
    peak_dep = float(slab_dep_s.max())
    peak_jac = float(slab_jac_s.max())
    n_det = n_detected(sim, detector_label)

    # Absolute-peak report for the figure caption. The two panels are
    # normalized to their own peaks for display, but the absolute levels
    # tell the "rarity of detection" story — expert caption should quote
    # both peaks and the ratio.
    ratio = (peak_dep / peak_jac) if peak_jac > 0 else float("inf")
    print(
        f"[fig 6] N_launched={sim.n_launched}  n_det({detector_label})={n_det}\n"
        f"        peak Phi (mm/launched-photon/column) = {peak_dep:.4g}\n"
        f"        peak J   (mm/detected-photon/column) = {peak_jac:.4g}\n"
        f"        Phi_peak / J_peak                    = {ratio:.2f}x"
    )

    xedges, zedges = bins[0], bins[2]
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    zcenters = 0.5 * (zedges[:-1] + zedges[1:])

    # Zoom to the source+detector region with a pad, extra headroom above
    # the scalp so the source marker and its "S" label are fully visible.
    active_mask = (slab_dep_s >= 0.01 * peak_dep) | (slab_jac_s >= 0.01 * peak_jac)
    if active_mask.any():
        x_active = xcenters[active_mask.any(axis=1)]
        z_active = zcenters[active_mask.any(axis=0)]
        xlim = (max(xedges[0], x_active.min() - 15),
                min(xedges[-1], x_active.max() + 15))
        zlim = (max(zedges[0], z_active.min() - 15),
                min(zedges[-1], z_active.max() + 8))
    else:
        xlim, zlim = (xedges[0], xedges[-1]), (zedges[0], zedges[-1])

    # Scalp outline in the slab: take mesh vertices with |y - y_src| < slab
    # and draw as a thin gray scatter.  Gives anatomical context without
    # needing a separate mesh-slicing step.
    try:
        scalp_solid = sim.scene.getSolid("brain_scalp")
        verts = np.array([v.array for v in scalp_solid.vertices])
        sm = np.abs(verts[:, 1] - sim.source_y) < slab_half_thickness_mm
        scalp_x, scalp_z = verts[sm, 0], verts[sm, 2]
    except Exception:
        scalp_x = scalp_z = None

    # ---- plotting ----
    import matplotlib.patheffects as pe
    fig = plt.figure(figsize=(8.6, 4.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[30, 1],
                           hspace=0.35, wspace=0.10,
                           left=0.08, right=0.98, top=0.88, bottom=0.14)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    cax = fig.add_subplot(gs[1, :])

    levels = np.linspace(-4, 0, 33)
    cmap = plt.get_cmap("magma")
    text_halo = [pe.withStroke(linewidth=2.5, foreground="black")]
    marker_color = "#19d2e6"   # bright cyan — readable on magma at both ends
    last_cs = None
    for ax, slab, peak, title in [
        (ax1, slab_dep_s, peak_dep, "(a) all photons"),
        (ax2, slab_jac_s, peak_jac, "(b) detected photons"),
    ]:
        if peak <= 0:
            ax.set_title(title + " — no data", fontsize=11)
            continue
        log_norm = np.log10(np.maximum(slab / peak, 10 ** levels[0]))
        last_cs = ax.contourf(xcenters, zcenters, log_norm.T,
                               levels=levels, cmap=cmap)
        if scalp_x is not None and len(scalp_x) > 0:
            ax.scatter(scalp_x, scalp_z, s=0.25, c="white",
                        alpha=0.25, edgecolors="none", zorder=2)
        d_xy = sim.detector_positions[detector_label]
        ax.plot(sim.source_pos[0], sim.source_pos[2], "o",
                 color=marker_color, ms=9, mec="k", mew=1.5, zorder=10)
        ax.plot(d_xy[0], d_xy[2], "s",
                 color=marker_color, ms=9, mec="k", mew=1.5, zorder=10)
        txt_s = ax.annotate("S", (sim.source_pos[0], sim.source_pos[2]),
                             xytext=(7, -14), textcoords="offset points",
                             color="white", fontsize=11, fontweight="bold",
                             zorder=11)
        txt_s.set_path_effects(text_halo)
        txt_d = ax.annotate("D", (d_xy[0], d_xy[2]),
                             xytext=(7, -14), textcoords="offset points",
                             color="white", fontsize=11, fontweight="bold",
                             zorder=11)
        txt_d.set_path_effects(text_halo)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$x$ (mm)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    ax1.set_ylabel(r"$z$ (mm)", fontsize=10)
    plt.setp(ax2.get_yticklabels(), visible=False)

    if last_cs is not None:
        cbar = fig.colorbar(last_cs, cax=cax, orientation="horizontal")
        cbar.set_label(r"$\log_{10}(\text{pathlength density}\,/\,\text{peak})$",
                        fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # Figure title announces the slab position so the reader knows which
    # coronal plane they are looking at.
    fig.suptitle(
        rf"Sagittal slab at $y = {sim.source_y:.0f} \pm {slab_half_thickness_mm:g}$ mm "
        rf"(source $y_{{\mathrm{{src}}}}$)",
        fontsize=11, y=0.97,
    )

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext, dpi in [("pdf", 300), ("png", 200)]:
        path = os.path.join(out_dir, f"fig4b_deposition_vs_jacobian.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {path}")


# --------------------------------------------------------------------------- #
# Figure 5 - sensitive volume on the Jacobian slice
# --------------------------------------------------------------------------- #
# Generic diffuse-optics sensitive-volume presentation (Patterson-Chance-Wilson
# 1989 analytical-banana tradition, Arridge 1999 review Section 3): filled
# region where J >= 0.01 * J_peak (the "sensitive volume", V_01) with an inner
# contour where J >= 0.50 * J_peak (the "core" half-max volume, V_50). Both
# volumes are reported in mm^3 from the 3D Jacobian. Same quantity applies to
# any turbid medium; the Colin27 mesh is incidental.

def figure5_sensitive_volume(sim: SimResult, bin_mm: float = 0.5,
                              slab_half_thickness_mm: float = 2.0):
    print("\n[fig 5] Sensitive volume (J >= 1% peak) and core (J >= 50% peak) ...")
    xyz_limits = sim.scene.getBoundingBox().xyzLimits
    limits = (tuple(xyz_limits[0]), tuple(xyz_limits[1]), tuple(xyz_limits[2]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    bin_volume = bin_mm ** 3

    header = f"  {'sep':<6} {'n_det':>6} {'V_50 (mm^3)':>13} {'V_01 (mm^3)':>13} {'peak J':>10}"
    print("\n" + header)
    print("  " + "-" * len(header))

    for ax, det_label in zip(axes, DETECTOR_LABELS):
        hist3d, bins = rasterise_jacobian(sim, det_label, bin_mm, limits)
        sep = det_label.split("_")[1]
        n_det = n_detected(sim, det_label)

        if hist3d.size == 0 or hist3d.max() <= 0:
            print(f"  {sep:<6} {n_det:>6}  no detections")
            ax.set_title(f"{sep}: no detections")
            continue

        peak = float(hist3d.max())
        # Generic sensitive volumes (any turbid medium): bins at >=1% and >=50% of peak
        v01 = float(np.sum(hist3d >= 0.01 * peak)) * bin_volume
        v50 = float(np.sum(hist3d >= 0.50 * peak)) * bin_volume
        print(f"  {sep:<6} {n_det:>6} {v50:>13.1f} {v01:>13.1f} {peak:>10.3g}")

        # Sagittal slab for display
        yedges = bins[1]
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        ymask = np.abs(ycenters - sim.source_y) <= slab_half_thickness_mm
        if not ymask.any():
            ymask = np.zeros_like(ycenters, dtype=bool)
            ymask[np.argmin(np.abs(ycenters - sim.source_y))] = True
        slab = hist3d[:, ymask, :].max(axis=1)  # max projection so outline is well defined

        xedges, zedges = bins[0], bins[2]
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        zcenters = 0.5 * (zedges[:-1] + zedges[1:])

        # Faint grey background of log Jacobian (orientation only)
        bg = np.log10(np.maximum(slab / peak, 1e-5)).T
        ax.imshow(
            bg, origin="lower",
            extent=(xedges[0], xedges[-1], zedges[0], zedges[-1]),
            aspect="equal", cmap="Greys", vmin=-5, vmax=0, alpha=0.4,
        )
        # Filled "sensitive volume" region J >= 1% peak
        mask01 = (slab >= 0.01 * peak).T
        ax.contourf(xcenters, zcenters, mask01.astype(float),
                    levels=[0.5, 1.5], colors=["#fdae61"], alpha=0.6)
        # Inner core J >= 50% peak (outline only)
        cs50 = ax.contour(xcenters, zcenters, slab.T,
                          levels=[0.50 * peak], colors=["#d7301f"], linewidths=2.0)
        cs01 = ax.contour(xcenters, zcenters, slab.T,
                          levels=[0.01 * peak], colors=["#fdae61"], linewidths=1.5)

        ax.plot(sim.source_pos[0], sim.source_pos[2], "r*", ms=14, mec="k", mew=1.0)
        d_xy = sim.detector_positions[det_label]
        ax.plot(d_xy[0], d_xy[2], "wo", ms=9, mec="k", mew=1.0)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z (mm)")
        ax.set_title(
            f"{sep}  |  V_01 = {v01:.0f} mm$^3$,  V_50 = {v50:.0f} mm$^3$"
        )
        proxies = [
            plt.Line2D([0], [0], color="#d7301f", lw=2),
            plt.Line2D([0], [0], color="#fdae61", lw=1.5),
        ]
        ax.legend(proxies,
                  [r"core ($J \geq 0.5\, J_{peak}$)",
                   r"sensitive vol. ($J \geq 0.01\, J_{peak}$)"],
                  loc="lower left", fontsize=8)

    fig.suptitle(
        "Sensitive volume (generic diffuse-optics convention, Patterson 1989 / "
        "Arridge 1999)"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(run_options: Optional[List[int]] = None):
    if run_options is None:
        run_options = [1, 2, 3, 4, 5, 6]

    sim = simulate()
    for det_label in DETECTOR_LABELS:
        print(f"[info] {det_label}: {n_detected(sim, det_label)} detected photons")

    if 1 in run_options:
        figure1_sagittal_jacobian(sim)
    if 2 in run_options:
        figure2_depth_profile(sim)
    if 3 in run_options:
        figure3_per_tissue(sim)
    if 4 in run_options:
        figure4_summary_table(sim)
    if 5 in run_options:
        figure5_sensitive_volume(sim)
    if 6 in run_options:
        figure6_deposition_vs_jacobian(sim)

    save_dir = os.environ.get("SAMPLING_VOL_SAVE_DIR")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for idx, num in enumerate(plt.get_fignums(), start=1):
            f = plt.figure(num)
            path = os.path.join(save_dir, f"jacobian_fig{idx}.png")
            f.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[save] {path}")
    else:
        plt.show()


if __name__ == "__main__":
    opts = None
    if len(sys.argv) > 1:
        opts = [int(a) for a in sys.argv[1:]]
    main(run_options=opts)
