"""
Figure 4 – Photon transport in the Colin27 anatomical brain model.

Loads the Colin27 brain atlas mesh (Fang 2010) directly from .mat format.
The mesh provides 4 nested surfaces (scalp, CSF, gray matter, white matter)
totaling 119,554 triangles and 70,226 vertices.

A divergent fiber-optic source on the scalp launches 10^6 photons into the
tissue. The simulation produces the characteristic banana-shaped photon
migration pattern between source and detector positions.

Based on: pytissuemodelsDEC2025_edema/PyTissueOptics/research/misc/runColin27_simple.py
"""

import numpy as np
import matplotlib.pyplot as plt

from pytissueoptics import (
    Vector, DivergentSource,
    ScatteringMaterial, ScatteringScene,
    EnergyLogger, Viewer,
    View2DProjectionX, View2DProjectionY, View2DProjectionZ,
    View2DSliceX, View2DSliceY, View2DSliceZ,
    hardwareAccelerationIsAvailable,
)
from pytissueoptics.scene.geometry import Environment, SurfaceCollection, Triangle, Vertex
from pytissueoptics.scene.solids import Solid, Cuboid

try:
    import scipy.io
except ImportError:
    raise ImportError("scipy required: pip install scipy")


TITLE = "Photon transport in the Colin27 anatomical brain model"

DESCRIPTION = """ Monte Carlo simulation of photon propagation through the Colin27 brain atlas.
The segmented mesh provides surfaces for scalp, CSF, gray matter, and white matter.
A fiber-optic source on the scalp produces the characteristic banana-shaped migration pattern. """

# Mesh path (from Fang Q. MCX project)
MESH_PATH = "/Users/williamboucher/Documents/pytissuemodelsDEC2025/PyTissueOptics/pytissueoptics/examples/benchmarks/MMC_Collins_Atlas_Mesh_Version_2L.mat"

# Optical properties at 830nm (standard fNIRS wavelength)
# The original mesh combines scalp+skull into one layer.
# Values from Okada & Delpy 2003, Strangman 2003, Yaroslavsky 2002, Jacques 2013
WAVELENGTH = 830  # nm
OPTICAL_PROPERTIES = {
    'world': ScatteringMaterial(mu_s=0, mu_a=0, g=1.0, n=1.0),
    'scalp': ScatteringMaterial(mu_s=8.5, mu_a=0.016, g=0.91, n=1.45),   # Combined scalp+skull
    'csf': ScatteringMaterial(mu_s=0.01, mu_a=0.004, g=0.90, n=1.33),
    'gray': ScatteringMaterial(mu_s=10.5, mu_a=0.032, g=0.90, n=1.40),
    'white': ScatteringMaterial(mu_s=41.0, mu_a=0.012, g=0.85, n=1.40),
}

# Detector material (transparent)
DETECTOR_MATERIAL = ScatteringMaterial(mu_s=0, mu_a=0, g=1.0, n=1.0)


def compute_vertex_normals(vertices, faces):
    """Compute area-weighted outward-pointing vertex normals."""
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
    """Load the Colin27 mesh and return the scene with all tissue layers."""
    data = scipy.io.loadmat(MESH_PATH)
    nodes = data["node"]
    faces = data["face"].copy()

    # Convert to 0-based indexing
    faces[:, :3] -= 1
    faces[:, 3] -= 1  # Surface IDs: 0=scalp, 1=CSF, 2=gray, 3=white

    layer_config = [
        ("scalp", 0, 'scalp', 'world'),
        ("csf", 1, 'csf', 'scalp'),
        ("grayMatter", 2, 'gray', 'csf'),
        ("whiteMatter", 3, 'white', 'gray'),
    ]

    solids = []
    scalp_vertices = None
    scalp_normals = None

    total_polygons = 0
    for label, surface_id, inside_mat, outside_mat in layer_config:
        mask = faces[:, 3] == surface_id
        surface_faces = faces[mask][:, :3]
        vertex_indices = np.unique(surface_faces.flatten())
        n_tris = mask.sum()
        total_polygons += n_tris

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
            material=OPTICAL_PROPERTIES[inside_mat]
        )

        if surface_id == 0:
            solid.setOutsideEnvironment(Environment(material=OPTICAL_PROPERTIES[outside_mat]))
        else:
            solid.setOutsideEnvironment(Environment(material=OPTICAL_PROPERTIES[outside_mat], solid=solids[-1]))

        solids.append(solid)
        print(f"  {label}: {n_tris} triangles, {len(vertex_indices)} vertices")

    scene = ScatteringScene(solids=solids, ignoreIntersections=True)

    return scene, scalp_vertices, scalp_normals, total_polygons


def find_scalp_point(scalp_vertices, scalp_normals, x, y, z_min=100):
    """Find the closest scalp surface point near (x, y) on the upper head."""
    mask = scalp_vertices[:, 2] > z_min
    verts = scalp_vertices[mask]
    norms = scalp_normals[mask]

    dists = np.sqrt((verts[:, 0] - x)**2 + (verts[:, 1] - y)**2)
    idx = dists.argmin()
    return tuple(verts[idx]), tuple(norms[idx])


def create_flat_detector(position, normal, size=3.0, thickness=0.5, label="detector"):
    """Create a flat cuboid detector on the scalp surface."""
    offset_pos = Vector(
        position[0] - normal[0] * (thickness / 2 + 0.5),
        position[1] - normal[1] * (thickness / 2 + 0.5),
        position[2] - normal[2] * (thickness / 2 + 0.5)
    )
    return Cuboid(
        a=size, b=size, c=thickness,
        material=DETECTOR_MATERIAL,
        position=offset_pos,
        label=label
    )


def exampleCode():
    N = 1000000 if hardwareAccelerationIsAvailable() else 10000

    # Load Colin27 brain atlas
    print("\nLoading Colin27 brain atlas mesh...")
    scene, scalp_vertices, scalp_normals, total_polygons = load_colin27()

    print(f"\n  Total mesh: {total_polygons} triangles")

    # Print optical properties table
    print(f"\n  Optical properties at {WAVELENGTH}nm:")
    print(f"  {'Layer':<15} {'μs (mm⁻¹)':>10} {'μa (mm⁻¹)':>10} {'g':>6} {'n':>6}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*6} {'-'*6}")
    for name in ['scalp', 'csf', 'gray', 'white']:
        mat = OPTICAL_PROPERTIES[name]
        print(f"  {name:<15} {mat.mu_s:>10.2f} {mat.mu_a:>10.4f} {mat.g:>6.2f} {mat.n:>6.2f}")

    # Find source position on top of head (motor cortex region)
    source_x, source_y = 90, 105
    scalp_pos, scalp_normal = find_scalp_point(scalp_vertices, scalp_normals, source_x, source_y)

    print(f"\n  Source on scalp: ({scalp_pos[0]:.1f}, {scalp_pos[1]:.1f}, {scalp_pos[2]:.1f}) mm")

    # Source pointing into head (along inward normal), placed 2mm above surface
    source_pos = Vector(
        scalp_pos[0] - scalp_normal[0] * 2,
        scalp_pos[1] - scalp_normal[1] * 2,
        scalp_pos[2] - scalp_normal[2] * 2
    )
    source_dir = Vector(scalp_normal[0], scalp_normal[1], scalp_normal[2])

    source = DivergentSource(
        position=source_pos,
        direction=source_dir,
        diameter=0.8,      # 800μm fiber core
        divergence=0.22,   # NA = 0.22
        N=N,
        displaySize=5
    )

    # Add detectors at typical fNIRS separations
    detector_separations = [20, 30, 40]  # mm
    print(f"\n  Detectors at separations: {detector_separations} mm")
    for sep in detector_separations:
        det_pos, det_normal = find_scalp_point(scalp_vertices, scalp_normals,
                                                source_x + sep, source_y)
        if det_pos is not None:
            detector = create_flat_detector(det_pos, det_normal,
                                            label=f"detector_{sep}mm")
            detector.setOutsideEnvironment(scene.getWorldEnvironment())
            scene.add(detector)
            print(f"    {sep}mm: ({det_pos[0]:.1f}, {det_pos[1]:.1f}, {det_pos[2]:.1f}) mm")

    # Preview geometry
    scene.show(source=source)

    # Run simulation
    print(f"\n  Propagating {N:,} photons...")
    logger = EnergyLogger(scene)
    source.propagate(scene, logger)

    viewer = Viewer(scene, source, logger)
    viewer.reportStats()

    # Visualizations
    viewer.show2D(View2DProjectionX())
    viewer.show2D(View2DProjectionZ())
    viewer.show2D(View2DSliceY(position=source_y))
    viewer.show3D()


if __name__ == "__main__":
    exampleCode()
