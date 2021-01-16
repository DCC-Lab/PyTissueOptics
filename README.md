# Monte Carlo in Python

This is an extremely simple object-oriented code in Python that simulates the propagation of light in tissue. It is not just *simple*: it is **outrageously simple** and **very slow** (see below). However, it is **extremely easy to understand** and most importantly **very simple modify**.

It may be slow, but speed is more than code performance: anyone can simulate something instantly instead of having to understand C, C++ or god forbid GPU code.  Therefore, you can modifiy everything in a day, get your result in a few hours instead of learning C (a few weeks), learn to work with compiled code (on separate machines) and modify C code written by someone else (weeks?). I think the overall speed to be concerned about is "the time it takes to get an answer", not necessarily the time it takes to run 100,000 photons. Considering many calculations take much less than an hour, it is fairly reasonable to imagine you could start a calculation in Python and run it overnight and get an answer the next day. I think there is a need for such a solution, and you will find it here.

## Getting started

You can run the example program:

```shell
python montecarlo.py
```

It will show you a graph of the energy deposited in the plane xz:

<img src="README.assets/image-20201014234533031.png" alt="image-20201014234533031" style="zoom:50%;" />

## Limitations

There are many limitations, as this is mostly a teaching tool but I have used it for real calculations:
1. There are no layer and/or interfaces : it is in a simple infinite volume (for now).
2. It only uses Henyey-Greenstein
3. It does not compute all stats yet.
4. Documentation is sparse at best.
5. Did I say it was slow? It is 200x slower than the well-known code [MCML](https://omlc.org/software/mc/mcml/) on the same machine.

## Advantages

However, there are advantages:

1. It is extremely simple to understand.
2. The code is very clear, and only a few files.
3. Can be used for teaching tissue optics.
4. It can be used as a good example of object-oriented programming for those not familiar with it.
5. It is relatively easy to modify for your own purpose.
6. In addition, it is very easy to parallelize a Monte Carlo calculations, because all runs are independant, so splitting the job onto several CPUs is a good option to gain a factor of close to 10 in perfromance on many computers.  

The code is in fact so simple, here is a complete calculation that takes 10 seconds on my computer:

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

