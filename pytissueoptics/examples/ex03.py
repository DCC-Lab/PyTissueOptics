from pytissueoptics.rayscattering import PencilSource, Stats
from pytissueoptics.scene import Logger

TITLE = "Propagate in a custom scene"

DESCRIPTION = """  
There is a Cuboid(), a Sphere() and an Ellipsoid(). They all go into a RayScatteringScene which takes a list of solid.
We can use the MayaviViewer to view our scene before propagation. Then, we repeat the usual steps of propagation.
"""


def exampleCode():
    from pytissueoptics.scene import Vector, Cuboid, Sphere, Ellipsoid, MayaviViewer
    from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
    from pytissueoptics.rayscattering import ScatteringMaterial

    myMaterial1 = ScatteringMaterial(mu_s=1.0, mu_a=1.0, g=0.7)
    myMaterial2 = ScatteringMaterial(mu_s=5.0, mu_a=0.5, g=0.8)

    cuboid = Cuboid(a=1, b=3, c=1, position=Vector(2, 0, 0), material=myMaterial1)
    sphere = Sphere(radius=0.5, position=Vector(0, 0, 0), material=myMaterial2)
    ellipsoid = Ellipsoid(a=1, b=3, c=2, position=Vector(-2, 0, 0), material=myMaterial1)
    myCustomScene = RayScatteringScene([cuboid, sphere, ellipsoid], worldMaterial=myMaterial2)

    # viewer = MayaviViewer()
    # viewer.addScene(myCustomScene)
    # viewer.show()

    logger = Logger()
    source = PencilSource(position=Vector(-3, 0, 0), direction=Vector(1, 0, 0), N=100)
    source.propagate(myCustomScene, logger)

    stats = Stats(logger, source, myCustomScene)
    stats.showEnergy3D()


if __name__ == "__main__":
    exampleCode()
