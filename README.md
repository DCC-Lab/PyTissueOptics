# Monte Carlo in Python

This is simple-object-oriented code that simulates the propagation of light in tissue. It is fairly simple and fairly slow, but it is very easy to understand and modify.



## Getting started



```shell
python montecarlo.py
```

will show you a graph of the energy deposited in the plane xz

<img src="README.assets/image-20201014234533031.png" alt="image-20201014234533031" style="zoom:50%;" />

The code is fairly simple:

```python
import numpy as np
from vector import *
from material import *
from photon import *

import time

if __name__ == "__main__":
    N = 1000 # number of photons
    mat = Material(mu_s=30, mu_a = 0.5, g = 0.8) # material
    
    try:
	      # continue previous calculation
        mat.stats.restore("output.json") 
    except:
        # start over if file does not exist
        mat.stats = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))

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

