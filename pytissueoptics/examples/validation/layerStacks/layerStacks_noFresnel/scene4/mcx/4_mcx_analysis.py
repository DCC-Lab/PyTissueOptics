import jdata as jd

data = jd.load("./4.jnii")
print(data["NIFTIData"])
data = data["NIFTIData"]

import numpy as np


def find_voxel_indices_in_layer(grid_size, voxel_size, layerPosition, layerthickness):
    """ Calculate the enscribed energy in a semi infinite layer (up to the bounds of the domain),
     of thickness layerthickness, at position layerPosition. "the position describes the begining of the layer"""
    indices = []

    # Calculate the bounding box for the layer to minimize checks
    x_min = 0
    x_max = grid_size
    y_min = 0
    y_max = grid_size
    z_min = layerPosition
    z_max = layerPosition + layerthickness

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            for z in range(z_min, z_max):
                indices.append((x, y, z))

    return indices

layer1 = np.sum([data[x, y, z] for x, y, z in find_voxel_indices_in_layer(1000, 0.01, 0, 10)])
layer2 = np.sum([data[x, y, z] for x, y, z in find_voxel_indices_in_layer(1000, 0.01, 10, 1)])
layer3 = np.sum([data[x, y, z] for x, y, z in find_voxel_indices_in_layer(1000, 0.01, 11, 10)])
a = np.asarray(data).shape
print("\n========================================")
print(f"Layer 1 Absorbed Energy: {layer1}")
print(f"Layer 2 Absorbed Energy: {layer2}")
print(f"Layer 3 Absorbed Energy: {layer3}")
print("========================================")
print(f"Total Absorbed Energy: {layer1 + layer2 + layer3}", "total energy in the domain: ", np.sum(data))
print("========================================")
