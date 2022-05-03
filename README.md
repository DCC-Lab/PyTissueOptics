# Light Propagation with Monte Carlo in Python

This python package is an object-oriented implementation of Monte Carlo Light Propagation simulation in diffuse media. The package is **extremely easy to use**, but as it is in python, it is **very slow** compared to C++ alternatives. (Parrallel-acceleration coming)

However, as discussed in the [why use this package](#why-use-this-package) section, code efficiency isn't the only variable at play. it is **extremely easy to understand**, **easily scalable** and **very simple to modify** for your need. It was designed with research and education in mind.

## Getting started

Install with `pip` or get the [code](https://github.com/DCC-Lab/PyTissueOptics) from GitHub.

```shell
pip install pytissueoptics
```

## What it can do

There are 6 main concepts (or `Class` in object-oriented language) in this code:

1. `Photon`: The photon is the main actor:  it has a position, it propagates in a given direction.  Its direction is changed when it scatters. It does not know anything about geometry or physical properties of tissue.
2. `Source`: A group of photons, such as `IsotropicSource`, `PencilSource` with specific properties. You provide the characteristics you want and it will give you a list of photons that responds to these criteria.  This list of photons will give you the answer you want after it has propagated in the `Object` of interest.
3. `Material`: The scattering properties and the methods to calculate the scattering angles are the responsibility of the `Material` class. The `Material` also knows how to move the photon between two points (for instance, if there is birefringence, this is where you would put it although polarization is currently not implemented).
4. `Geometry`: A real-world geometry (`Box`, `Cube`, `Sphere`, `Layer`, etc...). The geometry has two important variables: a `Material` (which will dictate its optical properties i.e. scattering and index) and a `Stats` object to keep track of physical values of interest.  The material will provide the required functions to compute the scattering angles, and the photon will use these angles to compute its next position.  The `Stats` object will compute the relevant statistics.
5. `Stats`: An object to keep track of something you want. For now, it keeps track of volumetric quantities (i.e. the energy deposited in the tissue) and intensities through the surfaces delimiting geometries.
6. Finally, very useful `Vector`, `UnitVector` and `Surface` helper classes with their subclasses are used to simplify any 3D computation with vectors, planes, surfaces, because they can be used like other values (they can be added, subtracted, normalized, etc...).

## How to go about modifying for your own purpose

1. Maybe you have a special light source?

   1. Subclass `Source` with your own light source and compute the photon properties in `newPhoton` according to your own rules. Use your source instead of `IsotropicSource` in the example above:

      ```python
      class MySource(Source):
           def __init__(self, myProperty, position, maxCount):
               super(MySource, self).__init__(position, maxCount)
               self.myProperty = myProperty
               
           def newPhoton(self) -> Photon:
               p = Photon()
               # Do your thing here with self.myProperty and modify p
               return p
            
      ```

2. Maybe you have a special material scattering  model?

   1. Subclass `Material` and override the methods for `getScatteringAngles()`.  Use your material in your geometry instead of `Material` in the example above. You could use the photon direction, polarization, position, or even its wavelength to compute its scattering angles:

      ```python
      class FunkyMaterial(Material):
           def __init__(self, myProperty, mu_s = 0, mu_a = 0, g = 0):
               super(MySource, self).__init__(mu_s, mu_a, g)
               self.myProperty = myProperty
               
           def getScatteringAngles(self, photon) -> (float, float):
               # Do your thing here with self.myProperty and compute theta, phi
               # Use Photon if needed (position, direction, wavelength, etc..)
               return (theta, phi)
            
      ```

3. If `photon.keepPathHistory()` is called, it will keep track all positions during its lifetime. You can then compute whatever you want by rewriting that part of the code or with a hook function I will write at some point.

4. Maybe you want to compute some funky stats? At each step, `scoreInVolume` is called with the photon and the drop in energy at that step.  When leaving the geometry through a surface, `scoreWhenCrossing` is called with the photon and the last position inside.

5. Maybe your have a special geometry? Subclass `Geometry` and override the `contains` method to compute whether or not a given position is inside your object or not, and `intersection` to compute the point on the surface of your object where the photon exits.


## Why use this package
It may be slow, but speed is more than code performance: anyone with little to no experience can simulate something instantly instead of having to understand C, C++ or, GPU code. With this package, you can quickly build your simulation in minutes and get your results in a few hours, instead of learning C (a few weeks?), learn to work with compiled code (a few days? libraries anyone?) and finally modify the C code written by someone else (days? weeks?). Considering this, the overall speed to be concerned about is "the time it takes to get an answer", not necessarily "the time it takes to run 100,000 photons". It is fairly reasonable to imagine you could start a calculation in Python, run it overnight and get an answer the next day after a few hours of calculations. This is the solution that the CPU-based portion of this package offers you.

Therefore, the whole point is the following: this code is perfect for quickly prototyping a small calculation, and then determine if you need performance or not. For many things, you actually don't.

### Advantages
However, there are advantages:

1. It is extremely simple to understand and to use.
2. It can be used for teaching tissue optics.
3. It can be used for teaching object-oriented programming for those not familiar with it.
5. It is fairly easy to modify for your own purpose. Many modifications do not even require to subclass.

### Limitations
There are some limitations as of now.

1. It uses Henyey-Greenstein approximation for scattering direction because it is sufficient most of the time.
2. Reflections are specular, which does not accounts for the roughness of materials. It is planned to implement Bling-Phong reflection model in a future release.
2. It is approximately 50x slower than the well-known code [MCML](https://omlc.org/software/mc/mcml/) on the same machine. However, this will be accounted for when photons intersections will be coded in OpenCL in a future release.

## Acknowledgment
This package was first inspired by the standard, tested, and loved [MCML from Wang, Jacques and Zheng](https://omlc.org/software/mc/mcpubs/1995LWCMPBMcml.pdf) , itself based on [Prahl](https://omlc.org/~prahl/pubs/abs/prahl89.html) and completely documented, explained, dissected by [Jacques](https://omlc.org/software/mc/) and [Prahl](https://omlc.org/~prahl/pubs/abs/prahl89.html). This would not be possible without the work of these pionneers.