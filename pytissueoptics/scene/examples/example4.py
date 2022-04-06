TITLE = "Lidar Scene"

DESCRIPTION = """ A Lidar simulation is a common use of a light propagation engine. By using a UniformRaySource and
and logging the data, you can imitate the data points of a lidar.
"""


def exampleCode():
    from pytissueoptics.scene import Logger, MayaviViewer, Vector
    from pytissueoptics.scene.intersection import FastIntersectionFinder, UniformRaySource
    from pytissueoptics.scene.tests.scene.benchmarkScenes import PhantomScene

    scene = PhantomScene()
    logger = Logger()
    intersectionFinder = FastIntersectionFinder(scene)
    source = UniformRaySource(position=Vector(0, 5, 0), direction=Vector(0, 0, 1), xTheta=180, yTheta=90, xResolution=1024, yResolution=512)

    for ray in source.rays:
        intersection = intersectionFinder.findIntersection(ray)
        if not intersection:
            continue
        logger.logPoint(intersection.position)

    viewer = MayaviViewer()
    viewer.addLogger(logger, pointScale=0.025)
    viewer.show()


if __name__ == "__main__":
    exampleCode()
