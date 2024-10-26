import jdata as jd

data = jd.load("./sphshells.jnii")
print(data["NIFTIData"])
data = data["NIFTIData"]

import numpy as np


def find_voxel_indices_in_sphere(grid_size, voxel_size, center, radius):
    """Calculate voxel indices for a sphere within a 3D grid."""
    indices = []

    # Calculate the bounding box for the sphere to minimize checks
    x_min = max(int((center[0] - radius) / voxel_size), 0)
    x_max = min(int((center[0] + radius) / voxel_size) + 1, grid_size)
    y_min = max(int((center[1] - radius) / voxel_size), 0)
    y_max = min(int((center[1] + radius) / voxel_size) + 1, grid_size)
    z_min = max(int((center[2] - radius) / voxel_size), 0)
    z_max = min(int((center[2] + radius) / voxel_size) + 1, grid_size)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                voxel_center = np.array([x * voxel_size + 0.5 * voxel_size,
                                         y * voxel_size + 0.5 * voxel_size,
                                         z * voxel_size + 0.5 * voxel_size])
                if np.linalg.norm(voxel_center - np.array(center)) <= radius:
                    indices.append((x, y, z))

    return indices


# Example usage
grid_size = 13  # Define the size of the grid as 60x60x60
voxel_size = 1  # Define the voxel size as 1x1x1 mm
center = [6.5, 6.5, 6.5]  # Sphere center in the grid

voxel_indices_outer = find_voxel_indices_in_sphere(grid_size, voxel_size, center, 5)
"try to remove only 1 voxel wide of the outershell to be sure that no grid voxel is included in the outer shell energy"
voxel_indices_outer_conservative = find_voxel_indices_in_sphere(grid_size, voxel_size, center, 4.9)
voxel_indices_core = find_voxel_indices_in_sphere(grid_size, voxel_size, center, 4.5)

# voxel_indices now contains the (x, y, z) indices of all voxels inside the sphere

energyDataOuter = np.sum([data[x, y, z] for x, y, z in voxel_indices_outer])
energyDataCore = np.sum([data[x, y, z] for x, y, z in voxel_indices_core])
energyDataGrid = np.sum(data)


energyGrid = energyDataGrid - energyDataOuter
energyDataOuter = energyDataOuter - energyDataCore



print("\n========================================")
print(f"Grid Absorbed Energy: {energyGrid}")
print(f"Outer Shell Absorbed Energy: {energyDataOuter}")

print(f"Core Absorbed Energy: {energyDataCore}")
print(f"total by summing all the energies: {energyGrid + energyDataOuter + energyDataCore}")
print("========================================")
print(f"Total Absorbed Energy: {energyDataGrid}")
print("========================================\n")

