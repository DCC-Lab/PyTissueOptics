
<h1 align="center"><b>PyTissueOptics</b></h1>

<p align="center"><i>Monte Carlo simulations of light transport made easy.</i></p>
<p align="center">
<img src="./docs/README.assets/pytissue-demo-banenr.jpg">
</p>

This python package is an object-oriented implementation of Monte Carlo modeling for light tranport in diffuse media. The package is **extremely easy to use**, and **polyvalent** as it allows simulations in arbitrary complex scenes. The package offers both a default Python implementation as well as an hardware-accelerated version using OpenCL. 

As discussed in the [why use this package](#why-use-this-package) section, code efficiency isn't the only variable at play. This code is **easy to understand**, **easily scalable** and **very simple to modify** for your need. It was designed with **research and education** in mind.

## Notable features
- Arbitrarily complex 3D environments.
- Import external 3D models (.OBJ).
- Great data visualization with `Mayavi`.
- Multi-layered tissues.
- Hardware acceleration with `OpenCL`.

## Getting started
Install with `pip` or get the [code](https://github.com/DCC-Lab/PyTissueOptics) from GitHub. If you're having trouble installing,
please [read the documentation](https://pytissueoptics.readthedocs.io/en/latest/).

```shell
pip install pytissueoptics
```
To launch a simple simulation, follow these steps.
1. Import the `pytissueoptics` module
2. Define the following objects:
    - `scene`: a `RayScatteringScene` object, which defines the scene and the optical properties of the media. This objects takes in a list of `Solid` as its argument. These `Solid` will have a `ScatteringMaterial` and a position. This is clear in the example code below.
    - `source`: a `Source` object, which defines the source of photons (`hardwareAccelerated=True` can be set here)
    - `logger`: a `Logger` object, which logs the simulation progress
    - `stats`: a `Stats` object, which computes the statistics of the simulation and displays the results
3. propagate the photons in your `scene` with `source.propagate`.
4. display the results by calling the appropriate method from your `stats` object.

Here's what it might look like:
```python
from pytissueoptics import *

material = ScatteringMaterial(mu_s=3.0, mu_a=1.0, g=0.8, n=1.5)

tissue = Cuboid(a=1, b=3, c=1, position=Vector(2, 0, 0), material=material)
scene = RayScatteringScene([tissue])

logger = Logger()
source = PencilPointSource(position=Vector(-3, 0, 0), direction=Vector(1, 0, 0), N=1000)
source.propagate(scene, logger)

stats = Stats(logger, source, scene)
stats.showEnergy3D()
stats.report()
```
For more details on how to use this package for your own research, please refer to the [documentation](https://pytissueoptics.readthedocs.io/en/latest/).

Also, you can check out the `pytissueoptics/examples` folder for more examples on how to use the package.


## Hardware acceleration
Hardware acceleration can offer a speed increase factor around 1000x depending on the scene. By default, the program will use the Python implementation, but hardware acceleration can be turned on when creating any light `Source`:
```python
source = DivergentSource(hardwareAccelerated=True, ...)
```
Follow the instructions on screen to get setup properly for the first execution. It will require the package PyOpenCL to be installed as well as OpenCL drivers for your hardware of choice. 

## Why use this package
It is known, as April of 2022, Python is **the most used** language ([Tiobe index](https://www.tiobe.com/tiobe-index/)).
This is due to the ease of use, the gentle learning curve, and growing community and tools. There was a need for 
such a package in Python, so that not only long hardened C/C++ programmers could use the power of Monte Carlo simulations.
It is fairly reasonable to imagine you could start a calculation in Python in a few minutes, run it overnight and get
an answer the next day after a few hours of calculations. It is also reasonable to think you could **modify** the code
yourself to suit your exact needs! (Do not attempt this in C). This is the solution that the CPU-based portion of this package 
offers you. With the new OpenCL implementation, speed is not an issue anymore, so using `pytissueoptics` should not even be a question.

### Known limitations
1. It uses Henyey-Greenstein approximation for scattering direction because it is sufficient most of the time.
2. Reflections are specular, which does not accounts for the roughness of materials. It is planned to implement Bling-Phong reflection model in a future release.

## Acknowledgment
This package was first inspired by the standard, tested, and loved [MCML from Wang, Jacques and Zheng](https://omlc.org/software/mc/mcpubs/1995LWCMPBMcml.pdf) , itself based on [Prahl](https://omlc.org/~prahl/pubs/abs/prahl89.html) and completely documented, explained, dissected by [Jacques](https://omlc.org/software/mc/) and [Prahl](https://omlc.org/~prahl/pubs/abs/prahl89.html). This would not be possible without the work of these pionneers.
