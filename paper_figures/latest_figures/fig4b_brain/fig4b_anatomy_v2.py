"""
Figure 4b (v2) -- Sampling volume with an anatomical 3D companion panel.

Three-panel "mock A" layout:

    [ (a) 3D cortex + sampling-volume photons ] [ (b) all photons ] [ (c) detected ]

Panel (a) is a real Mayavi render of the Colin27 cortical (gray-matter) surface -- visible
gyri/sulci -- with the translucent scalp and the DETECTED-photon sampling-volume cloud (the
"banana") shown in the actual anatomy. This carries the "complex 3D brain" point now that the
"triangle count = realism" wording is gone. Panels (b)/(c) are the existing sagittal-slab
pathlength-density maps (all photons vs detected). Source S (cyan) and detector D (orange) use
identical markers + labels in all three panels, and the 3D view is a coronal view (looking
along y) so its x-z axes match the slice panels.

Heavy lifting (mesh load, photon sim, rasterisation) is reused from
fig4b_sampling_volume_comparison.py. This script only adds the 3D anatomy render + composition.

Usage:
    LOW photon count for fast iteration (default):
        MPLBACKEND=Agg python fig4b_anatomy_v2.py
    Bump for the final figure:
        FIG4B_N=500000 python fig4b_anatomy_v2.py
    NO_3D=1 skips the Mayavi render (still writes the two slice panels).
"""
import os
import sys

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig4b_sampling_volume_comparison as f4b  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Prefer a local copy of the mesh if one has been dropped next to this script (the original
# fig4b path points outside the project and may be unreadable from a sandboxed session).
_LOCAL_MESH = os.path.join(OUT_DIR, "MMC_Collins_Atlas_Mesh_Version_2L.mat")
if os.path.exists(_LOCAL_MESH):
    f4b.MESH_PATH = _LOCAL_MESH
    print(f"[mesh] using local copy: {_LOCAL_MESH}")

# Low default photon count for brainstorming; bump via FIG4B_N for the final figure.
LOW_N = int(os.environ.get("FIG4B_N", 30_000))
f4b.N_PHOTONS_GPU = LOW_N
f4b.N_PHOTONS_CPU = LOW_N

DETECTOR_FOR_CLOUD = "detector_20mm"   # which channel's sampling volume to show in 3D

# Unified S/D styling across all panels.
S_COLOR = (0.0, 0.78, 0.86)            # cyan
D_COLOR = (0.95, 0.45, 0.0)            # orange
CLOUD_COLORMAP = "hot"

# 3D render appearance / camera (coronal: look along y, z up, x horizontal -> matches slices).
RENDER_SIZE = (820, 900)
CORTEX_COLOR = (0.82, 0.70, 0.66)
SCALP_COLOR = (0.95, 0.86, 0.78)
CORTEX_OPACITY = 0.9
SCALP_OPACITY = 0.12
CAM_AZIMUTH = -90.0                     # flip to +90 if S/D come out left-right mirrored
N_CLOUD_POINTS = 20_000                 # subsample of sampling-volume events drawn in 3D


# --------------------------------------------------------------------------- #
# Surface extraction for rendering (gray-matter cortex + scalp)
# --------------------------------------------------------------------------- #
def load_surface(surface_id):
    """Return (vertices[N,3], faces[M,3] 0-based) for one Colin27 surface id
    (0=scalp, 1=csf, 2=gray, 3=white). For rendering only."""
    data = scipy.io.loadmat(f4b.MESH_PATH)
    nodes = data["node"]
    faces = data["face"].copy()
    faces[:, :3] -= 1
    faces[:, 3] -= 1
    sf = faces[faces[:, 3] == surface_id][:, :3]
    vi = np.unique(sf)
    remap = {g: l for l, g in enumerate(vi)}
    local = np.array([[remap[v] for v in f] for f in sf])
    return nodes[vi], local


# --------------------------------------------------------------------------- #
# Detected-photon sampling-volume cloud (real points)
# --------------------------------------------------------------------------- #
def sampling_cloud(sim, detector_label, max_points=N_CLOUD_POINTS):
    """Return (xyz[K,3], logw[K]) of detected-photon pathlength contributions
    (the sampling volume), subsampled and log-scaled for display."""
    flog = f4b.filtered(sim, detector_label)
    per_event = f4b.pathlength_events(flog)
    if not per_event:
        return None, None
    allpts = np.concatenate(list(per_event.values()), axis=0)   # (n, 4): w, x, y, z
    w, xyz = allpts[:, 0], allpts[:, 1:4]
    if len(w) > max_points:
        idx = np.random.default_rng(0).choice(len(w), max_points, replace=False)
        w, xyz = w[idx], xyz[idx]
    logw = np.log10(np.maximum(w, w.max() * 1e-4))
    return xyz, logw


# --------------------------------------------------------------------------- #
# 3D anatomy render (Mayavi)
# --------------------------------------------------------------------------- #
def render_cortex_3d(sim, out_png):
    import mayavi.mlab as mlab

    gray_v, gray_f = load_surface(2)
    scalp_v, scalp_f = load_surface(0)
    xyz, logw = sampling_cloud(sim, DETECTOR_FOR_CLOUD)

    fig = mlab.figure(size=RENDER_SIZE, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
    mlab.triangular_mesh(gray_v[:, 0], gray_v[:, 1], gray_v[:, 2], gray_f,
                         color=CORTEX_COLOR, opacity=CORTEX_OPACITY)
    mlab.triangular_mesh(scalp_v[:, 0], scalp_v[:, 1], scalp_v[:, 2], scalp_f,
                         color=SCALP_COLOR, opacity=SCALP_OPACITY)

    if xyz is not None:
        pts = mlab.points3d(xyz[:, 0], xyz[:, 1], xyz[:, 2], logw, scale_mode="none",
                            scale_factor=2.2, mode="sphere", colormap=CLOUD_COLORMAP,
                            opacity=0.12, resolution=8)
        pts.module_manager.scalar_lut_manager.reverse_lut = True   # high density = dark/red

    # S/D markers + labels (sized in mm; head is ~140 mm so these read clearly).
    for pos, col, lab in [(sim.source_pos, S_COLOR, "S"),
                          (sim.detector_positions[DETECTOR_FOR_CLOUD], D_COLOR, "D")]:
        mlab.points3d([pos[0]], [pos[1]], [pos[2]], color=col, scale_factor=12,
                      resolution=20, mode="sphere")
        mlab.text3d(pos[0] + 5, pos[1], pos[2] + 10, lab, color=(0, 0, 0), scale=11)

    # Coronal view (look along y, +z up): parallel projection framed tightly to the head so
    # it fills the panel instead of floating in whitespace.
    center = scalp_v.mean(axis=0)
    aspect = RENDER_SIZE[0] / RENDER_SIZE[1]
    x_ext = np.ptp(scalp_v[:, 0])
    z_ext = np.ptp(scalp_v[:, 2])
    # Zoom in a touch (0.90) and shift the focal point up toward S/D so the slightly tighter
    # frame keeps the optode/cortex region and only crops the brain base.
    parallel_scale = 0.90 * max(z_ext / 2, (x_ext / 2) / aspect)
    focal = (center[0], center[1], center[2] + 0.10 * z_ext)
    mlab.view(azimuth=CAM_AZIMUTH, elevation=90, focalpoint=focal)
    cam = fig.scene.camera
    cam.view_up = (0.0, 0.0, 1.0)
    cam.parallel_projection = True
    cam.parallel_scale = parallel_scale
    fig.scene.render()
    mlab.savefig(out_png, size=RENDER_SIZE, magnification=1)
    mlab.close(fig)
    _autocrop(out_png)
    print(f"[3D] saved {out_png}")


def _autocrop(path, pad=12):
    """Trim the white border so the head fills the image."""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        l, t, r, b = bbox
        l, t = max(0, l - pad), max(0, t - pad)
        r, b = min(im.width, r + pad), min(im.height, b + pad)
        im.crop((l, t, r, b)).save(path)


# --------------------------------------------------------------------------- #
# Sagittal slab panels (reuse fig4b rasterisation)
# --------------------------------------------------------------------------- #
def compute_slab(hist, bins, source_y, slab_half=3.0):
    yc = 0.5 * (bins[1][:-1] + bins[1][1:])
    ymask = np.abs(yc - source_y) <= slab_half
    if not ymask.any():
        ymask = np.zeros_like(yc, dtype=bool)
        ymask[np.argmin(np.abs(yc - source_y))] = True
    return hist[:, ymask, :].sum(axis=1)


def draw_slab(ax, slab, peak, bins, sim, title, smooth_sigma_mm=1.5, bin_mm=1.0):
    xc = 0.5 * (bins[0][:-1] + bins[0][1:])
    zc = 0.5 * (bins[2][:-1] + bins[2][1:])
    slab_s = gaussian_filter(slab, sigma=smooth_sigma_mm / bin_mm)
    levels = np.linspace(-4, 0, 33)
    cmap = plt.get_cmap("magma")
    ax.set_facecolor(cmap(0.0))   # zero-density background = colormap floor (no white gap)
    cs = None
    if peak > 0:
        log_norm = np.log10(np.maximum(slab_s / peak, 10 ** levels[0]))
        cs = ax.contourf(xc, zc, log_norm.T, levels=levels, cmap=cmap, extend="both")
    d = sim.detector_positions[DETECTOR_FOR_CLOUD]
    _marker(ax, sim.source_pos[0], sim.source_pos[2], S_COLOR, "o", "S")
    _marker(ax, d[0], d[2], D_COLOR, "s", "D")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$x$ (mm)", fontsize=10)
    ax.tick_params(labelsize=9)
    return cs


def _marker(ax, x, z, color, shape, label):
    halo = [pe.withStroke(linewidth=2.5, foreground="black")]
    ax.plot(x, z, shape, color=color, ms=9, mec="k", mew=1.5, zorder=10)
    t = ax.annotate(label, (x, z), xytext=(7, -14), textcoords="offset points",
                    color="white", fontsize=11, fontweight="bold", zorder=11)
    t.set_path_effects(halo)


# --------------------------------------------------------------------------- #
# Main: render 3D, compute slabs, compose 3-panel figure
# --------------------------------------------------------------------------- #
def main():
    print(f"[fig4b-v2] N = {LOW_N} photons (set FIG4B_N to change)")
    sim = f4b.simulate()
    for d in f4b.DETECTOR_LABELS:
        print(f"  {d}: {f4b.n_detected(sim, d)} detected")

    png_3d = os.path.join(OUT_DIR, "fig4b_cortex_photons.png")
    have_3d = False
    if os.environ.get("NO_3D") != "1":
        try:
            render_cortex_3d(sim, png_3d)
            have_3d = os.path.exists(png_3d)
        except Exception as e:
            print(f"  [3D] render skipped/failed ({type(e).__name__}: {e}); slices still produced.")

    # Slab maps (all photons vs detected), same recipe as fig4b figure6.
    xyz_limits = sim.scene.getBoundingBox().xyzLimits
    limits = (tuple(xyz_limits[0]), tuple(xyz_limits[1]), tuple(xyz_limits[2]))
    bin_mm = 1.0
    hist_dep, bins = f4b.rasterise_deposition(sim, bin_mm, limits)
    hist_jac, _ = f4b.rasterise_jacobian(sim, DETECTOR_FOR_CLOUD, bin_mm, limits)
    slab_dep = compute_slab(hist_dep, bins, sim.source_y)
    slab_jac = compute_slab(hist_jac, bins, sim.source_y)
    peak_dep = gaussian_filter(slab_dep, 1.5).max()
    peak_jac = gaussian_filter(slab_jac, 1.5).max()

    # Zoom window to the active region.
    xc = 0.5 * (bins[0][:-1] + bins[0][1:])
    zc = 0.5 * (bins[2][:-1] + bins[2][1:])
    act = (gaussian_filter(slab_dep, 1.5) >= 0.01 * peak_dep)
    if act.any():
        xa, za = xc[act.any(axis=1)], zc[act.any(axis=0)]
        # Cap the top just above the optodes so there is no empty band above the scalp.
        z_top = max(za.max(), sim.source_pos[2],
                    sim.detector_positions[DETECTOR_FOR_CLOUD][2]) + 1.0
        xlim = (xa.min() - 12, xa.max() + 12)
        zlim = (za.min() - 12, z_top)
    else:
        xlim = zlim = None

    # ---- Compose 3-panel figure ----
    fig = plt.figure(figsize=(14, 4.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.0, 1.0],
                          height_ratios=[30, 1], hspace=0.32, wspace=0.10,
                          left=0.04, right=0.98, top=0.9, bottom=0.13)
    ax3d = fig.add_subplot(gs[0, 0])
    axa = fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[0, 2], sharey=axa)
    cax = fig.add_subplot(gs[1, 1:])

    if have_3d:
        ax3d.imshow(mpimg.imread(png_3d))
    else:
        ax3d.text(0.5, 0.5, "(3D render unavailable)", ha="center", va="center")
    ax3d.axis("off")
    ax3d.set_title("(a) Cortical surface + sampling volume", fontsize=11)

    draw_slab(axa, slab_dep, peak_dep, bins, sim, "(b) All photons")
    cs = draw_slab(axb, slab_jac, peak_jac, bins, sim, "(c) Detected photons")
    axa.set_ylabel(r"$z$ (mm)", fontsize=10)
    plt.setp(axb.get_yticklabels(), visible=False)
    if xlim:
        for ax in (axa, axb):
            ax.set_xlim(*xlim)
            ax.set_ylim(*zlim)

    if cs is not None:
        cbar = fig.colorbar(cs, cax=cax, orientation="horizontal")
        cbar.set_label(r"$\log_{10}(\text{pathlength density}\,/\,\text{peak})$", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    for ext, dpi in [("pdf", 300), ("png", 200)]:
        path = os.path.join(OUT_DIR, f"fig4b_anatomy_v2.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[save] {path}")


if __name__ == "__main__":
    main()
