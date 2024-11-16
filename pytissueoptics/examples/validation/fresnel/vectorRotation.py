import numpy as np
from pytissueoptics.scene.geometry.vector import Vector
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ScatterableObject:
    def __init__(self, scattering_method='pto'):
        self.direction = Vector(0, 1, 1)
        self.anyOrthogonal = self.direction.getAnyOrthogonal()
        self.anyOrthogonal.normalize()
        self.direction.normalize()
        self.g = 0.9  # Set the anisotropy parameter
        self.scattering_method = scattering_method

    def getScatteringAngles(self):
        phi = np.random.random() * 2 * np.pi  # Azimuthal angle
        g = self.g
        if g == 0:
            cost = 2 * np.random.random() - 1
        else:
            temp = (1 - g * g) / (1 - g + 2 * g * np.random.random())
            cost = (1 + g * g - temp * temp) / (2 * g)
        theta = np.arccos(cost)
        return theta, phi

    def pto_scatter(self):
        theta, phi = self.getScatteringAngles()
        self.anyOrthogonal.rotateAround(self.direction, phi)
        self.direction.rotateAround(self.anyOrthogonal, theta)
        return theta, phi

    def mcml_scatter(self):
        theta, phi = self.getScatteringAngles()
        self.direction.spin(theta, phi)
        return theta, phi

    def scatter(self):
        # Choose the scattering method based on the attribute
        if self.scattering_method == 'pto':
            return self.pto_scatter()
        elif self.scattering_method == 'mcml':
            return self.mcml_scatter()
        elif self.scattering_method == 'rodriguez':
            return self.rodriguez_scatter()
        else:
            raise ValueError(f"Unknown scattering method: {self.scattering_method}")

    # Optional additional scatter methods
    def rodriguez_scatter(self):
        pass


class RotationValidater:
    def __init__(self, methods):
        self.methods = methods
        self.scatter_points = {method: [] for method in methods}
        self.angles = {method: [] for method in methods}
        self.raw_angles = []
        self.raw_polar_angles = []

    def sample_raw_angles(self, n=1000):
        for _ in range(n):
            scatterableObject = ScatterableObject()
            theta, phi = scatterableObject.getScatteringAngles()

            # Convert (theta, phi) to Cartesian coordinates for 3D plotting
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            self.raw_angles.append((x, y, z))

            # Store raw (theta, phi) values for polar plot
            self.raw_polar_angles.append((theta, phi))

    def show_distribution_on_sphere(self, n=1000):
        # Sample raw angles for the initial distribution plot
        self.sample_raw_angles(n)

        # For each method, perform scattering and collect data
        for method in self.methods:
            for _ in range(n):
                scatterableObject = ScatterableObject(scattering_method=method)
                theta, phi = scatterableObject.scatter()

                # Record the final direction after scattering
                x, y, z = scatterableObject.direction.x, scatterableObject.direction.y, scatterableObject.direction.z
                self.scatter_points[method].append((x, y, z))

                # Store the scattering angles (theta, phi) for 2D projection
                self.angles[method].append((theta, phi))

        # Set up subplots with the raw angle distribution as the first plot
        fig = plt.figure(figsize=(8 * (len(self.methods) + 1), 8))
        grid_spec = fig.add_gridspec(1, len(self.methods) + 1, height_ratios=[1])

        # Plot each scattering method in its own subplot
        for idx, method in enumerate(self.methods):
            # 3D Scatter plot
            ax3d = fig.add_subplot(grid_spec[0, idx + 1], projection='3d')
            xs, ys, zs = zip(*self.scatter_points[method])  # Unpack x, y, z coordinates
            ax3d.scatter(xs, ys, zs, color='b', s=1, alpha=0.5, label=f"{method} Scattering")

            # Sphere wireframe outline
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 50)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))
            ax3d.plot_wireframe(x, y, z, color="gray", alpha=0.3, linewidth=0.5)

            # Set equal aspect ratio
            ax3d.set_box_aspect([1, 1, 1])
            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.set_title(f"{method} Scattering Distribution")

        plt.tight_layout()
        plt.show()


# Usage example: test both "pto" and "mcml" scattering methods
validator = RotationValidater(methods=["pto", "mcml"])
validator.show_distribution_on_sphere(n=1000)
