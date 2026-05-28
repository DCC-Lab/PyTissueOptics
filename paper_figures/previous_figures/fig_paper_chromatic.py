"""Publication figure: through-focus R90 per wavelength (axial chromatic aberration).

Single panel, paired with fig_paper_sa.py.  Same lens (f=100 mm, d=25.4 mm,
t=3.6 mm) re-simulated three times with the refractive index set from the
Sellmeier equation for Schott N-BK7 at λ = 450, 550, 650 nm.  Each wavelength
produces its own V-shaped R90(z) curve with a sharp minimum; the horizontal
separation between minima is the axial chromatic aberration.

Why a fixed lens geometry matters
---------------------------------
`SymmetricLens(f, material)` SOLVES the lensmaker equation for the sphere
radius R given (f, n).  If you call it with three different indices you get
three DIFFERENT physical lenses — each focuses at exactly f=100 mm in its own
n.  That cancels the chromatic shift we want to measure.

The fix: pick R once (from the 550-nm index) and instantiate a bare ThickLens
with that same R for every wavelength, varying only the material.  This is the
same thing you'd do at a bench: one lens, three monochromatic sources.

Thin-lens Cauchy reference
--------------------------
For a thin lens, Δf/f ≈ −Δn / (n − 1), so for the (450, 550, 650) nm triplet
of indices we predict a red-blue axial shift of ~+2.1 mm.  The measured value
(~+2.1 mm at 1M photons) matches to within ~3% even though the lens is
genuinely thick — remarkable robustness of the Cauchy approximation.

Environment variables
---------------------
N       : photons per wavelength (default 1M on GPU).
NO_3D=1 : skip interactive 3D viewers.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pytissueoptics import (
    Cuboid, DirectionalSource, EnergyLogger, ScatteringMaterial,
    ScatteringScene, Vector, hardwareAccelerationIsAvailable,
)
from pytissueoptics.scene.solids import ThickLens
from pytissueoptics.rayscattering.energyLogging import PointCloudFactory

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# 1M photons per wavelength gives ~1-2 µm R90 precision near best focus — more
# than enough to resolve the ~1 mm chromatic shift between adjacent lines.
N = int(os.environ.get("N", 1_000_000 if hardwareAccelerationIsAvailable() else 5_000))
SEED = 42

# Lens and beam: narrow 6 mm beam so spherical aberration stays small and
# doesn't dominate over the chromatic effect we're trying to see.
LENS_F, LENS_D, LENS_T, BEAM_D = 100.0, 25.4, 3.6, 6.0

# Screen grid: dense (0.1-0.3 mm) around z=99-101 where the three minima live.
Z_SCREENS = np.array([97.5, 98.25, 98.5, 98.7, 98.85, 99.0, 99.3, 99.7,
                      99.95, 100.1, 100.3, 100.55, 100.8, 101.0, 101.3])


def bk7(lam_um: float) -> float:
    """Schott N-BK7 refractive index from the 6-parameter Sellmeier equation.

    Takes wavelength in µm (so 0.450 for 450 nm) and returns n(λ).  The
    coefficients below are the published Schott N-BK7 datasheet values
    (accurate to ~1e-5 over the visible/NIR).
    """
    l2 = lam_um ** 2
    n2 = (1.0
          + 1.03961212 * l2 / (l2 - 0.00600069867)
          + 0.231792344 * l2 / (l2 - 0.0200179144)
          + 1.01046945 * l2 / (l2 - 103.560653))
    return float(np.sqrt(n2))


# Three wavelengths spanning the visible band: blue-green (F-line-ish),
# yellow-green (d-line), red (C-line-ish).  Colours chosen to be
# roughly spectrally appropriate so readers can intuit which curve is which.
WAVELENGTHS = [450.0, 550.0, 650.0]
COLOURS = {450.0: "#2166ac", 550.0: "#2ca02c", 650.0: "#d62728"}
N_AT = {lam: bk7(lam / 1000.0) for lam in WAVELENGTHS}

# ---------------------------------------------------------------------------
# 2. Fix the lens geometry once at the central wavelength
# ---------------------------------------------------------------------------
# Solve for the single radius of curvature that yields f=100 mm at the 550 nm
# index using the thick-lens lensmaker equation
#   1/f = (n−1)[2/R − (n−1)t/(nR^2)]
# rearranged into a quadratic for R.  This reproduces what
# SymmetricLens(f=..., n=N_REF) would give, but does it only once so every
# wavelength gets the SAME physical lens (different n, not different shape).
N_REF = N_AT[550.0]
p = np.sqrt(LENS_F * N_REF * (LENS_F * N_REF - LENS_T)) * (N_REF - 1) / N_REF
R = LENS_F * (N_REF - 1) + p


def run(n_value, keep_scene=False):
    """Run one MC sweep for a given glass index, return R90(z) along Z_SCREENS."""
    glass, vacuum = ScatteringMaterial(n=n_value), ScatteringMaterial()

    # Bare `ThickLens` (not `SymmetricLens`) so we can force our fixed R.
    # backRadius = -R makes the rear surface a mirror image of the front.
    lens = ThickLens(frontRadius=R, backRadius=-R, diameter=LENS_D,
                     thickness=LENS_T, material=glass, position=Vector(0, 0, 0))

    screens, labels = [], []
    for z in Z_SCREENS:
        lbl = f"s{z:.2f}"
        screens.append(Cuboid(30, 30, 0.05, position=Vector(0, 0, z + 0.025),
                              material=vacuum, label=lbl))
        labels.append(lbl)

    scene = ScatteringScene([lens, *screens])
    src = DirectionalSource(Vector(0, 0, -20), Vector(0, 0, 1), diameter=BEAM_D,
                            N=N, displaySize=5, seed=SEED)
    logger = EnergyLogger(scene)
    src.propagate(scene, logger, showProgress=False)

    # R90(z): the 90%-enclosed radius at each screen.  Standard metric for
    # spot size; robust to outliers, widely used in optical design.
    factory = PointCloudFactory(logger)
    r90 = np.full(len(Z_SCREENS), np.nan)
    for iz, lbl in enumerate(labels):
        pts = factory.getPointCloud(lbl, f"{lbl}_front").enteringSurfacePoints
        if pts is None or len(pts) == 0:
            continue
        r = np.hypot(pts[:, 1], pts[:, 2])
        w = np.abs(pts[:, 0])
        order = np.argsort(r)
        cum = np.cumsum(w[order])
        idx = int(np.searchsorted(cum, 0.9 * cum[-1]))
        r90[iz] = r[order][min(idx, len(r) - 1)]

    if keep_scene:
        return r90, scene, src, logger
    return r90


def refine_min(z, y):
    """Parabolic sub-screen refinement of argmin(y) vs z.

    Fit a parabola y = A(z−z*)² + C through the three R90 samples closest to
    the V-shape's minimum and solve for the vertex z*.  Handles unequal screen
    spacing (matters when the grid is non-uniform).  The clip at [z_lo, z_hi]
    protects against degenerate parabolas (A≤0 or collinear points).
    """
    i = int(np.nanargmin(y))
    if i == 0 or i == len(z) - 1:
        return float(z[i])   # argmin at edge -> just return the screen z
    x1, x2, x3 = z[i - 1], z[i], z[i + 1]
    y1, y2, y3 = y[i - 1], y[i], y[i + 1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0:
        return float(x2)
    # Lagrange-form quadratic coefficients (handles unequal spacing).
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3**2 * (y1 - y2) + x2**2 * (y3 - y1) + x1**2 * (y2 - y3)) / denom
    if A <= 0:
        return float(x2)   # concave-down — would extrapolate nonsense
    return float(np.clip(-B / (2 * A), x1, x3))


# ---------------------------------------------------------------------------
# 3. Run the three wavelengths
# ---------------------------------------------------------------------------
# Keep the 650-nm scene around for the optional 3D viewer at the end.
r90_per_lam = {}
scene_last = src_last = logger_last = None
for lam in WAVELENGTHS:
    if lam == WAVELENGTHS[-1]:
        r90_per_lam[lam], scene_last, src_last, logger_last = run(N_AT[lam], keep_scene=True)
    else:
        r90_per_lam[lam] = run(N_AT[lam])

# Sub-screen best-focus per wavelength.
z_best = {lam: refine_min(Z_SCREENS, r90_per_lam[lam]) for lam in WAVELENGTHS}
dz_rb = z_best[650.0] - z_best[450.0]

# Thin-lens Cauchy prediction: Δf/f ≈ −Δn/(n−1), evaluated between the 450 nm
# and 650 nm indices and referenced to the OBSERVED 550 nm focus (which is
# ~100.02 mm rather than exactly 100 because the lens is thick: back focal
# length is slightly different from paraxial f).
dz_pred = -z_best[550.0] * (N_AT[650.0] - N_AT[450.0]) / (N_AT[550.0] - 1.0)

print(f"  λ=450 nm: z_best = {z_best[450.0]:.3f} mm  (n = {N_AT[450.0]:.5f})")
print(f"  λ=550 nm: z_best = {z_best[550.0]:.3f} mm  (n = {N_AT[550.0]:.5f})")
print(f"  λ=650 nm: z_best = {z_best[650.0]:.3f} mm  (n = {N_AT[650.0]:.5f})")
print(f"  measured Δz(red-blue) = {dz_rb:+.3f} mm")
print(f"  thin-lens Δz(red-blue) = {dz_pred:+.3f} mm   (ratio = {dz_rb / dz_pred:.2f})")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.7, 3.1), constrained_layout=True)

for lam in WAVELENGTHS:
    # Main R90(z) curve per wavelength.
    ax.plot(Z_SCREENS, r90_per_lam[lam], "-o", ms=3.5, lw=1.4,
            color=COLOURS[lam], label=rf"$\lambda = {int(lam)}$ nm")
    # Vertical dashed line at that wavelength's parabolically-refined best
    # focus — makes the chromatic shift immediately visible.
    ax.axvline(z_best[lam], color=COLOURS[lam], ls="--", lw=0.9, alpha=0.7)

ax.set_xlabel(r"axial position $z$ (mm)", fontsize=10)
ax.set_ylabel(r"$R_{90}$ spot radius (mm)", fontsize=10)
# Log y: minima are ~5 µm and tails are ~100 µm, factor 20 — log is cleaner.
ax.set_yscale("log")
ax.set_xlim(97.5, 101.3)
ax.set_ylim(1e-3, 0.2)
ax.tick_params(axis="both", labelsize=9)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="upper right", framealpha=0.95,
          handletextpad=0.4, borderpad=0.3)

out = os.path.dirname(os.path.abspath(__file__))
for ext, dpi in [("pdf", 300), ("png", 200)]:
    path = os.path.join(out, f"fig_paper_chromatic.{ext}")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[save] {path}")

# ---------------------------------------------------------------------------
# 5. Optional 3D visualization
# ---------------------------------------------------------------------------
# Pops Mayavi windows for the last (650 nm) wavelength scene.  Skipped under
# NO_3D=1 or headless environments.
if scene_last is not None and os.environ.get("NO_3D") != "1":
    from pytissueoptics import Viewer, PointCloudStyle
    print("\n[3D] Showing setup at λ=650 nm (close window to continue) ...")
    scene_last.show(source=src_last)
    print("[3D] Showing post-propagation photon cloud (close window to exit) ...")
    Viewer(scene_last, src_last, logger_last).show3D(
        pointCloudStyle=PointCloudStyle(showSolidPoints=False))
