"""Figure 3 (v2) -- 4f relay: native-trace ray diagram + matched white-background 3D render.

Focused regeneration of the two 4f visuals (the 3D scene `4f_3d` and the meridional
ray diagram `fig3_4f_rays`) with:

  * NATIVE photon traces -- per-photon paths reconstructed by grouping the logger's
    raw events (getRawDataPoints, tagged with photonID). In a 4f relay a ray bends only
    at the two lenses, so its exact polyline is
        launch -> L1 front -> L1 back -> L2 front -> L2 back -> image,
    every vertex a real logged crossing. This replaces the old 21-screen scaffold with a
    single image-plane capture (needed only to terminate the converging segment).
  * 3D re-render on a WHITE background with propagation running LEFT -> RIGHT (parallel
    side view down the y-axis, optical axis horizontal), coloured by launch angle.
  * An optional COMBINED figure stacking the 3D render above the meridional diagram with a
    shared, aligned axial (z) axis.

Environment variables:
  N_RAY  : photons launched for the trace run (default 50k on GPU).
  NO_3D=1: skip the Mayavi render (still writes the 2D meridional figure).
"""
import os

import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from pytissueoptics import (
    Cuboid, DivergentSource, EnergyLogger, ScatteringMaterial,
    ScatteringScene, SymmetricLens, Vector, hardwareAccelerationIsAvailable,
)
from pytissueoptics.scene import get3DViewer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RAY = int(os.environ.get("N_RAY", 50_000 if hardwareAccelerationIsAvailable() else 2_000))
SEED = 42

LENS_F, LENS_D, LENS_T, GLASS_N = 50.0, 25.4, 3.6, 1.50
SOURCE_DIAM, SOURCE_DIV = 0.1, 0.3
Z_IMAGE = 3 * LENS_F                       # nominal image plane (z = 150 mm)

# Colour rays by signed x launch angle (coolwarm): +x rays red, -x rays blue.
RAY_CMAP = "coolwarm"
ANGLE_RANGE = 0.15                         # rad, colour-scale half-range

# --- 3D render (white bg, left-to-right side view, parallel projection) ---
BG_COLOR, FG_COLOR = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
LENS_RENDER_COLOR, LENS_RENDER_OPACITY = (0.35, 0.45, 0.75), 0.18
N_TRACES_3D = 260                          # rays drawn in the 3D bundle
RENDER_SIZE = (1850, 240)                  # very wide: the 4f axis is ~200 mm, aperture ~25 mm
CAM = dict(azimuth=90.0, elevation=90.0, roll=90.0, focalpoint=(0.0, 0.0, 50.0))
PARALLEL_SCALE = 15.0                      # half-height (mm, transverse) shown in the render
# Axial window shown == camera horizontal extent; reused for the 2D so panels line up.
_ZHALF = PARALLEL_SCALE * RENDER_SIZE[0] / RENDER_SIZE[1]
Z0, Z1 = CAM["focalpoint"][2] - _ZHALF, CAM["focalpoint"][2] + _ZHALF

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Scene + simulation
# ---------------------------------------------------------------------------
def build_scene():
    """Two identical lenses (f=50) + a single image-plane capture screen."""
    glass, vacuum = ScatteringMaterial(n=GLASS_N), ScatteringMaterial()
    lens1 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 0), label="Lens1")
    lens2 = SymmetricLens(f=LENS_F, diameter=LENS_D, thickness=LENS_T,
                          material=glass, position=Vector(0, 0, 2 * LENS_F), label="Lens2")
    image = Cuboid(40, 40, 0.1, position=Vector(0, 0, Z_IMAGE + 0.05),
                   material=vacuum, label="ImagePlane")
    scene = ScatteringScene([lens1, lens2, image])
    source = DivergentSource(position=Vector(0, 0, -LENS_F), direction=Vector(0, 0, 1),
                             N=N_RAY, diameter=SOURCE_DIAM, divergence=SOURCE_DIV,
                             displaySize=2, seed=SEED)
    return scene, source, (lens1, lens2)


def native_paths(logger):
    """Reconstruct per-photon polylines purely from native logged events.

    Everything is derived from the log (grouped by photonID), with no dependence on the
    launch-array ordering -- the GPU does not guarantee that a logged photonID indexes the
    source's launch arrays, so selecting "meridional" rays from those arrays is unreliable.
    The source is a known point at (0, 0, -f), so each ray's path is

        (0, 0, -f) -> L1 front -> L1 back -> L2 front -> L2 back -> image,

    every vertex after the first a real logged crossing. The launch angle theta_x and the
    meridional offset y are read from the ray's own first (L1-front) crossing.

    Returns {pid: (path[N, 3], theta_x, y_at_L1)} for photons that reach the image plane.
    """
    raw = logger.getRawDataPoints()                # (value, x, y, z, photonID)
    if raw is None or raw.shape[1] < 5:
        return {}
    ids = raw[:, 4].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    raw, ids = raw[order], ids[order]
    uniq, start = np.unique(ids, return_index=True)
    bounds = np.append(start, len(ids))

    out = {}
    for j, u in enumerate(uniq):
        pts = raw[bounds[j]:bounds[j + 1], 1:4]
        pts = pts[np.argsort(pts[:, 2])]           # order along the optical axis
        if pts[-1, 2] < Z_IMAGE - 10:              # never reached the image plane
            continue
        z_first = pts[0, 2]
        if not (-5.0 < z_first < 0.0):             # first crossing must be the L1 front face
            continue
        theta_x = pts[0, 0] / (z_first + LENS_F)   # launch angle from the source point
        path = np.vstack([[0.0, 0.0, -LENS_F], pts])
        out[int(u)] = (path, float(theta_x), float(pts[0, 1]))
    return out


# ---------------------------------------------------------------------------
# 2D meridional diagram
# ---------------------------------------------------------------------------
PLANE_LABELS = [(0, "L1", 10, "bold"), (2 * LENS_F, "L2", 10, "bold"),
                (-LENS_F, "source", 9, "normal"), (LENS_F, "Fourier", 9, "normal"),
                (Z_IMAGE, "image", 9, "normal")]


def draw_plane_labels(ax, y=1.03):
    """Place the key-plane labels just above `ax` (outside the data area)."""
    xtf = ax.get_xaxis_transform()
    for z_plane, txt, fs, w in PLANE_LABELS:
        ax.text(z_plane, y, txt, ha="center", va="bottom", fontsize=fs, weight=w,
                transform=xtf, clip_on=False)


def draw_meridional(ax, paths, show_xlabel=True, show_plane_labels=True):
    """Draw the native meridional ray polylines + lens/plane furniture onto `ax`.

    `paths` maps pid -> (path[N, 3], theta_x, y_at_L1).
    """
    cmap = plt.get_cmap(RAY_CMAP)

    ax.axvspan(-LENS_T / 2, LENS_T / 2, color="0.4", alpha=0.25, zorder=0)
    ax.axvspan(2 * LENS_F - LENS_T / 2, 2 * LENS_F + LENS_T / 2, color="0.4", alpha=0.25, zorder=0)
    for z_plane in (-LENS_F, LENS_F, Z_IMAGE):
        ax.axvline(z_plane, color="k", ls=":", lw=0.7, alpha=0.5, zorder=0)

    for pid in sorted(paths.keys(), key=lambda p: abs(paths[p][1])):
        arr, theta_x, _ = paths[pid]
        ax.plot(arr[:, 2], arr[:, 0], "-", color=cmap(0.5 + theta_x / (2 * ANGLE_RANGE)),
                lw=0.4, alpha=0.55, rasterized=True)

    if show_plane_labels:
        draw_plane_labels(ax)
    if show_xlabel:
        ax.set_xlabel(r"Axial position, $z$ (mm)", fontsize=11)
    ax.set_ylabel(r"Transverse position, $x$ (mm)", fontsize=11)
    ax.set_xlim(Z0, Z1)
    ax.set_ylim(-13, 13)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(False)


def add_angle_colorbar(fig, ax):
    sm = mcm.ScalarMappable(cmap=RAY_CMAP, norm=mcolors.Normalize(-ANGLE_RANGE, ANGLE_RANGE))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015, aspect=20)
    cbar.set_label(r"Launch angle, $\theta_x$ (rad)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    return cbar


# ---------------------------------------------------------------------------
# 3D render (white bg, left-to-right)
# ---------------------------------------------------------------------------
def render_3d(lenses, paths_3d, out_path, show):
    import mayavi.mlab as mlab

    fig = mlab.figure("4f_3d", size=RENDER_SIZE, bgcolor=BG_COLOR, fgcolor=FG_COLOR)
    viewer3D = get3DViewer()
    for lens in lenses:
        viewer3D.add(lens, representation="surface", opacity=LENS_RENDER_OPACITY,
                     color=LENS_RENDER_COLOR)

    # Colour each ray by its launch angle theta_x (scalar + colormap, vmin/vmax symmetric).
    for arr, theta_x, _ in paths_3d.values():
        s = np.full(arr.shape[0], theta_x)
        mlab.plot3d(arr[:, 0], arr[:, 1], arr[:, 2], s, tube_radius=None, line_width=1.1,
                    colormap=RAY_CMAP, vmin=-ANGLE_RANGE, vmax=ANGLE_RANGE)

    mlab.figure(fig, bgcolor=BG_COLOR, fgcolor=FG_COLOR)
    mlab.view(azimuth=CAM["azimuth"], elevation=CAM["elevation"], roll=CAM["roll"],
              focalpoint=CAM["focalpoint"], figure=fig)
    fig.scene.camera.parallel_projection = True
    fig.scene.camera.parallel_scale = PARALLEL_SCALE
    fig.scene.render()
    mlab.savefig(out_path, size=RENDER_SIZE, magnification=1)
    print(f"[save] {out_path}")
    if show:
        mlab.show()
    else:
        mlab.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    scene, source, lenses = build_scene()
    logger = EnergyLogger(scene)
    source.propagate(scene, logger, showProgress=False)

    # Reconstruct every ray that reaches the image plane, straight from the log.
    all_paths = native_paths(logger)
    print(f"  reconstructed {len(all_paths)} native ray paths reaching the image plane")
    rng = np.random.default_rng(1)

    # 2D meridional subset (|y at L1| small so the x-z projection is faithful),
    # stratified across launch angle theta_x so every colour is represented.
    MERIDIONAL_Y = 1.0   # mm
    meridional = {p: v for p, v in all_paths.items() if abs(v[2]) < MERIDIONAL_Y}
    bins = np.linspace(-ANGLE_RANGE - 0.01, ANGLE_RANGE + 0.01, 17)
    by_bin = {i: [] for i in range(len(bins) - 1)}
    for pid, v in meridional.items():
        b = int(np.searchsorted(bins, v[1]) - 1)
        if 0 <= b < len(bins) - 1:
            by_bin[b].append(pid)
    chosen_2d = []
    for b, members in by_bin.items():
        if members:
            chosen_2d.extend(rng.choice(members, min(15, len(members)), replace=False))
    paths_2d = {int(p): meridional[int(p)] for p in chosen_2d}
    print(f"  2D meridional: {len(paths_2d)} native ray paths")

    # 3D subset: a general bundle across all azimuths (not restricted to meridional).
    pids = list(all_paths.keys())
    keep = rng.choice(pids, min(N_TRACES_3D, len(pids)), replace=False)
    paths_3d = {int(p): all_paths[int(p)] for p in keep}
    print(f"  3D bundle: {len(paths_3d)} native ray paths")

    # --- Figure 1: standalone meridional diagram ---
    fig_r, ax = plt.subplots(figsize=(7.4, 3.0), constrained_layout=True)
    draw_meridional(ax, paths_2d)
    add_angle_colorbar(fig_r, ax)
    for ext, dpi in [("pdf", 300), ("png", 200)]:
        path = os.path.join(OUT_DIR, f"fig3_4f_rays_v2.{ext}")
        fig_r.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {path}")
    plt.close(fig_r)

    # --- Figure 2: standalone 3D render ---
    png_3d = os.path.join(OUT_DIR, "fig3_4f_3d_v2.png")
    rendered = False
    if os.environ.get("NO_3D") != "1":
        try:
            render_3d(lenses, paths_3d, png_3d, show=False)
            rendered = True
        except Exception as e:
            print(f"  3D render skipped/failed ({type(e).__name__}: {e}); 2D figure is unaffected.")

    # --- Figure 3: combined, axially aligned stack (3D on top, meridional below) ---
    if rendered and os.path.exists(png_3d):
        fig_c = plt.figure(figsize=(8.4, 5.2))
        # Panels share column 0 (identical width -> aligned z-axis); colorbar lives in
        # its own column spanning both rows so it cannot disturb that alignment.
        gs = fig_c.add_gridspec(2, 2, width_ratios=[1.0, 0.028], height_ratios=[1.0, 1.5],
                                hspace=0.14, wspace=0.02)
        ax_top = fig_c.add_subplot(gs[0, 0])
        ax_bot = fig_c.add_subplot(gs[1, 0], sharex=ax_top)
        cax = fig_c.add_subplot(gs[:, 1])

        img = plt.imread(png_3d)
        ax_top.imshow(img, extent=[Z0, Z1, -PARALLEL_SCALE, PARALLEL_SCALE], aspect="auto")
        ax_top.set_xlim(Z0, Z1)
        ax_top.set_yticks([])
        ax_top.set_ylabel("3D bundle", fontsize=11)
        ax_top.tick_params(labelbottom=False)
        draw_plane_labels(ax_top, y=1.06)          # labels once, above the top panel

        draw_meridional(ax_bot, paths_2d, show_plane_labels=False)

        sm = mcm.ScalarMappable(cmap=RAY_CMAP, norm=mcolors.Normalize(-ANGLE_RANGE, ANGLE_RANGE))
        sm.set_array([])
        cbar = fig_c.colorbar(sm, cax=cax)
        cbar.set_label(r"Launch angle, $\theta_x$ (rad)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        for ext, dpi in [("pdf", 300), ("png", 200)]:
            path = os.path.join(OUT_DIR, f"fig3_4f_combined_v2.{ext}")
            fig_c.savefig(path, dpi=dpi, bbox_inches="tight")
            print(f"[save] {path}")
        plt.close(fig_c)


if __name__ == "__main__":
    main()
