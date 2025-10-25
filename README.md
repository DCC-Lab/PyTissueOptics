<h1 align="center"><b>PyTissueOptics</b></h1>
<p align="center"><i>A hardware-accelerated Python module to simulate light transport in arbitrarily complex 3D media with ease.</i></p>
<img src="https://raw.githubusercontent.com/DCC-Lab/PyTissueOptics/main/assets/banner.png" />

[![Tests](https://github.com/DCC-Lab/pytissueoptics/actions/workflows/tests.yaml/badge.svg)](https://github.com/DCC-Lab/pytissueoptics/actions/workflows/tests.yaml) [![codecov](https://codecov.io/gh/DCC-Lab/pytissueoptics/branch/main/graph/badge.svg)](https://codecov.io/gh/DCC-Lab/pytissueoptics) [![CodeFactor](https://www.codefactor.io/repository/github/DCC-Lab/pytissueoptics/badge)](https://www.codefactor.io/repository/github/DCC-Lab/pytissueoptics) [![License](https://img.shields.io/github/license/DCC-Lab/pytissueoptics.svg)](LICENSE)

This python package is a fast and flexible implementation of Monte Carlo modeling for light transport in diffuse media. 
The package is **easy to set up and use**, and its mesh-based approach makes it a polyvalent tool to simulate 
light transport in **arbitrarily complex scenes**. The package offers both a native Python implementation 
and a **hardware-accelerated** version using OpenCL which supports most GPUs and CPUs. 

Designed with **research and education** in mind, the code aims to be clear, modular, and easy to extend for a wide range of applications.

## Notable features
- Supports arbitrarily complex **mesh-based** 3D environments.
- **Normal smoothing** for accurate modeling of curved surfaces like lenses.
- Per-photon data points of deposited energy, fluence rate and intersection events.
- **Hardware accelerated** with `OpenCL` using [PyOpenCL](https://github.com/inducer/pyopencl).
- Photon traces & detectors.
- Import **external 3D models** (`.OBJ`).
- Many 3D visualization options built with [Mayavi](https://github.com/enthought/mayavi).
- Low memory mode with auto-binning to 2D views.
- **Reusable graphics framework** to kickstart other raytracing projects like [SensorSim](https://github.com/JLBegin/SensorSim).

## Installation
Requires Python 3.9+ installed. 

> Currently, the `pip` version is outdated. We recommend installing the development version.
1. Clone the repository.
2. Create a virtual environment inside the repository with `python -m venv venv`.
3. Activate the virtual environment. 
    - On MacOS/Linux: `source venv/bin/activate`.
    - On Windows: `venv\Scripts\activate.bat`.
4. Upgrade `pip` with `pip install --upgrade pip`.
5. Install the package requirements with `python -m pip install -e .[dev]`.

## Getting started

A **command-line interface** is available to quicky run a simulation from our pool of examples:

```
python -m pytissueoptics --help
```

You can kick start your first simulation using one of our **pre-defined scene** under the `samples` module. 

```python
from pytissueoptics import *

# Define (scene, source, logger)
N = 500_000
scene = samples.PhantomTissue()
source = DivergentSource(
   position=Vector(0, 0, -0.1), direction=Vector(0, 0, 1), N=N, diameter=0.2, divergence=0.78
)
logger = EnergyLogger(scene)

# Run
source.propagate(scene, logger=logger)

# Stats & Visualizations
viewer = Viewer(scene, source, logger)
viewer.reportStats()

viewer.show2D(View2DProjectionX())
viewer.show2D(View2DProjectionX(solidLabel="middleLayer"))
viewer.show2D(View2DSurfaceZ(solidLabel="middleLayer", surfaceLabel="interface0"))
viewer.show1D(Direction.Z_POS)
viewer.show3D()
```
#### Expected output
```
Report of solid 'backLayer'
  Absorbance: 67.78% (10.53% of total power)
    Transmittance at 'backLayer_back': 22.4%
    Transmittance at 'interface0': 4.9%
    ...
```



![stack_visuals](https://user-images.githubusercontent.com/29587649/219904076-f52c850f-7e86-45a3-8e32-ac3e1fbed051.png)

#### Scene definition
Here is the explicit definition of the above scene sample. We recommend you look at other examples to get familiar with the API.
```python
materials = [
   ScatteringMaterial(mu_s=2, mu_a=1, g=0.8, n=1.4),
   ScatteringMaterial(mu_s=3, mu_a=1, g=0.8, n=1.7),
   ScatteringMaterial(mu_s=2, mu_a=1, g=0.8, n=1.4),
]
w = 3
frontLayer = Cuboid(a=w, b=w, c=0.75, material=materials[0], label="frontLayer")
middleLayer = Cuboid(a=w, b=w, c=0.5, material=materials[1], label="middleLayer")
backLayer = Cuboid(a=w, b=w, c=0.75, material=materials[2], label="backLayer")
layerStack = backLayer.stack(middleLayer, "front").stack(frontLayer, "front")
scene = ScatteringScene([layerStack])
```

#### Hardware acceleration

Depending on your platform and GPU, you might already have OpenCL drivers installed, which should work out of the box.
Run a PyTissueOptics simulation first to see your current status.

> Follow the instructions on screen to get setup properly. It will offer to run a benchmark test to determine the ideal number of work units for your hardware. 
For more help getting OpenCL to work, refer to [PyOpenCL's documentation](https://documen.tician.de/pyopencl/misc.html#enabling-access-to-cpus-and-gpus-via-py-opencl) on the matter. Note that you can disable hardware acceleration at any time with `disableOpenCL()` or by setting the environment variable `PTO_DISABLE_OPENCL=1`.

## Examples

All examples can be run using the CLI tool:

```
python -m pytissueoptics --list
python -m pytissueoptics --examples 1,2,3
```

1. [Scene sample](/pytissueoptics/examples/ex01.py)
2. [Infinite medium](/pytissueoptics/examples/ex02.py)
3. [Optical lens & saving progress](/pytissueoptics/examples/ex03.py)
4. [Custom layer stack](/pytissueoptics/examples/ex04.py)
5. [Sphere in cube](/pytissueoptics/examples/ex05.py)
6. [Sampling volume simulation](/pytissueoptics/examples/ex06.py)

---

### Known limitations

1. It uses Henyey-Greenstein approximation for scattering direction because it is sufficient most of the time.
2. Reflections are specular, which does not account for the roughness of materials. A Bling-Phong reflection model could be added in a future release.

## Acknowledgment

This package was first inspired by the standard, tested, and loved [MCML from Wang, Jacques and Zheng](https://omlc.org/software/mc/mcpubs/1995LWCMPBMcml.pdf) , itself based on [Prahl](https://omlc.org/~prahl/pubs/abs/prahl89.html) and completely documented, explained, dissected by [Jacques](https://omlc.org/software/mc/) and [Prahl](https://omlc.org/~prahl/pubs/abs/prahl89.html). The original idea of using Monte Carlo for tissue optics calculations was [first proposed by Wilson and Adam in 1983](https://doi.org/10.1118/1.595361).  This would not be possible without the work of these pioneers.
