import numpy as np
from matplotlib import pyplot as plt

g = 0.9
f = lambda x: (1 / 4*np.pi) * (1 - g**2) / (1 + g**2 - 2*g*np.cos(x))**(3/2)
x = np.linspace(-np.pi, np.pi, 1000)

for g in [0, 0.3, 0.5, 0.8, 0.9, 0.95]:
    y = f(x)
    plt.semilogy(x, y, label=f"g={g}")

plt.xlabel("Scattering Angle (radians)")
plt.title("Probability Density Function of Scattering Angle")
plt.ylabel("P(θ)")
plt.legend()
plt.show()
