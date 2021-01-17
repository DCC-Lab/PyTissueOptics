# Light propagation with Monte Carlo in Python

This is an extremely simple object-oriented code in Python that simulates the propagation of light in scattering tissue. It is not just *simple*: it is **outrageously simple** and **very slow** (see below). However, it is **extremely easy to understand** and most importantly **very simple modify**.

It may be slow, but speed is more than code performance: anyone with little to no experience can simulate something instantly instead of having to understand C, C++ or, god forbid, GPU code.  Therefore, you can quickly modifiy everything in an afternoon and get your results in a few hours, instead of learning C (a few weeks?), learn to work with compiled code (a few days? libraries anyone?) and finally modify the C code written by someone else (days? weeks?). I think the overall speed to be concerned about is "the time it takes to get an answer", not necessarily "the time it takes to run 100,000 photons". Considering many calculations with high performance code (in C for instance) take a few minutes, it is fairly reasonable to imagine you could start a calculation in Python, run it overnight and get an answer the next day after a few hours of calculations. I think there is a need for such a solution, and you will find it here.

## Getting started

Download the code (it is not a Python module yet).  Go to the main directory where you can run the example program:

```shell
python montecarlo.py
```

You need Python 3, it will not work with Python 2. The example code will show you a graph of the energy deposited in the plane xz from a isotropic source at the origin:

<img src="README.assets/image-20210116103556173.png" alt="image-20210116103556173" style="zoom:50%;" />

Then it will display the logarithm (`log10`) of the intensity as a fonction of distance along the x direction:

<img src="README.assets/image-20210116104020740.png" alt="image-20210116104020740" style="zoom:50%;" />

Then, the idea would be to modify the code for your geometry (layers, boxes, cubes, spheres, etc...) and compute what you want.



## Limitations

There are many limitations, as this is mostly a teaching tool but I have used it for real calculations:
1. There are 3D objects, but reflections at the interface at not considered yet: the index of refraction for everything is `n=1`.
2. It only uses Henyey-Greenstein because it is sufficient most of the time.
3. It does not compute all possible stats (total reflectance, etc...), only the depositted energy in the volume.
4. Documentation is sparse at best.
5. You have probably noticed that the axes on the graphs are currently not labelled. Don't tell my students.
6. Did I say it was slow? It is approximately 200x slower than the well-known code [MCML](https://omlc.org/software/mc/mcml/) on the same machine. I know, and now I know that *you* know, but see **Advantages** below.

## Advantages

However, there are advantages:

1. It is extremely simple to understand.
2. The code is very clear with only a few files in a single directory.
3. It can be used for teaching tissue optics.
4. It can be used for teaching object-oriented programming for those not familiar with it.
5. It is fairly easy to modify for your own purpose.
6. In addition, because it is very easy to parallelize a Monte Carlo calculations (all runs are independant), splitting the job onto several CPUs is a good option to gain a factor of close to 10 in perfromance on many computers.

The code is in fact so simple, here is the complete code that created the above two graphs in 10 seconds on my computer:

```python
import numpy as np
from vector import *
from material import *
from photon import *

import time

if __name__ == "__main__":
    N = 1000 # number of photons
    mat = Material(mu_s=30, mu_a = 0.5, g = 0.8) # material is infinite
    
    try:
	# continue previous calculation
        mat.stats.restore("output.json") 
    except:
        # start over if file does not exist
        mat.stats = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41)) # volume over which we keep stats

    startTime = time.time()
    for i in range(1,N+1):
        photon = Photon()
        while photon.isAlive:
            d = mat.getScatteringDistance(photon)
            (theta, phi) = mat.getScatteringAngles(photon)
            photon.scatterBy(theta, phi)
            photon.moveBy(d)
            mat.absorbEnergy(photon)
            photon.roulette()
        if i  % 100 == 0:
            print("Photon {0}/{1}".format(i,N) )
            if mat.stats is not None:
                mat.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i))

    elapsed = time.time() - startTime
    print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed/N*1000, N))

    if mat.stats is not None:
        mat.stats.save("output.json")
        mat.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(N), realtime=False)
        mat.stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)

```

