import jdata as jd

data = jd.load("./cubeshells.jnii")
print(data["NIFTIData"])
data = data["NIFTIData"]

import numpy as np


def find_voxel_indices_in_cube(grid_size, voxel_size, center, sideLength):
    """Calculate voxel indices for a cube within a 3D grid."""
    indices = []

    # Calculate the bounding box for the cube to minimize checks
    x_min = max(int((center[0] - sideLength / 2) / voxel_size), 0)
    x_max = min(int((center[0] + sideLength / 2) / voxel_size) + 1, grid_size)
    y_min = max(int((center[1] - sideLength / 2) / voxel_size), 0)
    y_max = min(int((center[1] + sideLength / 2) / voxel_size) + 1, grid_size)
    z_min = max(int((center[2] - sideLength / 2) / voxel_size), 0)
    z_max = min(int((center[2] + sideLength / 2) / voxel_size) + 1, grid_size)

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                indices.append((x, y, z))

    return indices


# Example usage
grid_size = 60  # Define the size of the grid as 60x60x60
voxel_size = 1  # Define the voxel size as 1x1x1 mm
center = [30, 30, 30]  # Sphere center in the grid

voxel_indices_outer = find_voxel_indices_in_cube(grid_size, voxel_size, center, 50)
voxel_indices_inner = find_voxel_indices_in_cube(grid_size, voxel_size, center, 46)
voxel_indices_core = find_voxel_indices_in_cube(grid_size, voxel_size, center, 20)

# voxel_indices now contains the (x, y, z) indices of all voxels inside the sphere

energyDataOuter = np.sum([data[x, y, z] for x, y, z in voxel_indices_outer])
energyDataInner = np.sum([data[x, y, z] for x, y, z in voxel_indices_inner])
energyDataCore = np.sum([data[x, y, z] for x, y, z in voxel_indices_core])
energyDataGrid = np.sum(data)

energyGrid = energyDataGrid - energyDataOuter
energyDataOuter = energyDataOuter - energyDataInner
energyDataInner = energyDataInner - energyDataCore

print("\n========================================")
print(f"Grid Absorbed Energy: {energyGrid}")
print(f"Outer Shell Absorbed Energy: {energyDataOuter}")
print(f"Inner Shell Absorbed Energy: {energyDataInner}")
print(f"Core Absorbed Energy: {energyDataCore}")
print(f"Sum of Absorbed Energy: {energyGrid + energyDataOuter + energyDataInner + energyDataCore}")
print("========================================")
print(f"Total Absorbed Energy: {energyDataGrid}")
print("========================================")

