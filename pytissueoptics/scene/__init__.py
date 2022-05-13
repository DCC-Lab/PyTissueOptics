"""A simple Python-based 3D Engine with hardware-acceleration support.
https://github.com/DCC-Lab/PytissueOptics/scene

This package is intended to be used as a base for developing 3D applications in Python.
It provides a simple 3D scene creation and rendering engine. The IntersectionFinder class
allows you to find the intersection of a ray with a scene and use the intersection for your own purposes.
You can also inject your own `Material` implementation and manage interaction based on the material properties.

Create a `Scene` object and add different `Solid` objects to it.
Create an `IntersectionFinder` object and pass the `Scene` object to it, so it has a reference to all the `Polygon`.
You can then ask the `IntersectionFinder` to find the intersection of a ray with the scene by passing a `Ray` as such:
 `IntersectionFinder().findIntersection(ray)`.
Use the `Logger` object to log `Point`, `DataPoint` or `Segment` information.
Display the scene and the log data using the `Viewer` object, which is based on `Mayavi`.

Here's an example of how to use this package:
```
from pytissueoptics.scene import *


solid = Cuboid(a=1, b=2, c=3, position=Vector(0, 0, 1), material=Material()))
scene = Scene([mySolid])
intersectionFinder = FastIntersectionFinder(myScene)
logger = Logger()

aRay = Ray(origin=Vector(0, 0, 0), direction=Vector(0, 0, 1))
intersection = intersectionFinder.findIntersection(aRay)
logger.logDataPoint(intersection.position, intersection.distance, key=InteractionKey(None, None)))

viewer = Viewer(scene, logger)
viewer.show()

```

To help you naviguate the features, here's a class diagram of the entire package::
"""

from .solids import Cuboid, Cube, Sphere, Ellipsoid, Cylinder, Cone
from .geometry import Vector
from .scene import Scene
from .loader import Loader, loadSolid
from .viewer import MayaviViewer, MAYAVI_AVAILABLE
from .logger import Logger
