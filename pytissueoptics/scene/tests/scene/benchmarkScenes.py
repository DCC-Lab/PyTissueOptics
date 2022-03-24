from pytissueoptics.scene.scene import Scene
from pytissueoptics.scene.geometry import Vector, Polygon, SurfaceCollection, BoundingBox
from pytissueoptics.scene.solids import Cuboid, Sphere, Cube, Ellipsoid, Solid


class AAxisAlignedPolygonScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        vertices = [Vector(0, 0, 0), Vector(1, 0, 0), Vector(1, 1, 0)]
        polygon = Polygon(vertices=vertices)
        aSolid = Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=SurfaceCollection({"lonely": [polygon]}))
        aSolid._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([aSolid])


class APolygonScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        vertices = [Vector(0, 0, 0), Vector(1, 0, 1), Vector(-0.2, -0.5, 3)]
        polygon = Polygon(vertices=vertices)
        aSolid = Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=SurfaceCollection({"lonely": [polygon]}))
        aSolid._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([aSolid])


class ACubeScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(0, 0, 0))
        cuboid1._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([cuboid1])


class TwoCubesScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(0, 0, 0))
        cuboid1._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        cuboid2 = Cuboid(a=2, b=1, c=1, position=Vector(6, 1, 1))
        self._solids.extend([cuboid1, cuboid2])


class ASphereScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere = Sphere(position=Vector(0, 0, 0), order=1)
        sphere._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([sphere])


class TwoSpheresScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere = Sphere(position=Vector(0, 0, 0), order=0)
        ellipsoid = Ellipsoid(a=1, b=2, c=1, position=Vector(2, 2, 6), order=1)
        sphere._bbox = BoundingBox([-10, 10], [-10, 10], [-10, 10])
        self._solids.extend([sphere, ellipsoid])


class PhantomScene(Scene):
    ROOM = []
    CROSSWALK = []
    OBJECTS = []
    SIGN = []

    def __init__(self):
        super().__init__()
        self._create()
        self._solids = [*self.ROOM, *self.CROSSWALK, *self.OBJECTS, *self.SIGN]

    def _create(self):
        self.ROOM = self._room()
        self.CROSSWALK = self._crossWalk()
        self.OBJECTS = self._objects()
        self.SIGN = self._sign()

    def _room(self):
        w, d, h, t = 20, 20, 8, 0.1
        floor = Cuboid(w + t, t, d + t, position=Vector(0, -t / 2, 0))
        leftWall = Cuboid(t, h, d, position=Vector(-w / 2, h / 2, 0))
        rightWall = Cuboid(t, h, d, position=Vector(w / 2, h / 2, 0))
        backWall = Cuboid(w, h, t, position=Vector(0, h / 2, -d / 2))
        return [floor, leftWall, rightWall, backWall]

    def _crossWalk(self):
        return [Cuboid(0.7, 0.001, 4, position=Vector(i, 0, -8)) for i in range(-5, 5)]

    def _objects(self):
        cubeA = Cube(3, position=Vector(-5, 3 / 2, -6))
        cubeB = Cube(3, position=Vector(5, 3 / 2, -6))
        cubeB.rotate(0, 20, 0)
        cubeC = Cube(1, position=Vector(-5, 3.866, -6))
        cubeC.rotate(0, 0, 45)
        cubeC.rotate(45, 0, 0)
        sphere = Sphere(0.75, order=2, position=Vector(5, 3.75, -6))
        return [cubeA, cubeB, cubeC, sphere]

    def _sign(self):
        sign = Cuboid(1.5, 1.5, 0.001, position=Vector(7.8, 5, -5 + (0.1 + 0.01) / 2))
        sign.rotate(0, 0, 45)
        stand = Cuboid(0.1, 5, 0.1, position=Vector(7.8, 2.5, -5))
        return [sign, stand]


class RandomShapesScene(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        cuboid1 = Cuboid(a=1, b=3, c=1, position=Vector(4, -2, 6))
        cuboid2 = Cuboid(1, 1, 1, position=Vector(-2, -2, 0))
        sphere1 = Sphere(position=Vector(3, 3, 3), order=1)
        ellipsoid1 = Ellipsoid(1, 2, 3, position=Vector(10, 3, -3), order=2)
        self._solids.extend([cuboid1, cuboid2, sphere1, ellipsoid1])


class XAlignedSpheres(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere1 = Sphere(position=Vector(2, 0, 0), order=3)
        sphere2 = Sphere(position=Vector(5, 0, 0), order=3)
        sphere3 = Sphere(position=Vector(9, 0, 0), order=3)
        sphere4 = Sphere(position=Vector(20, 0, 0), order=3)
        self._solids.extend([sphere1, sphere2, sphere3, sphere4])


class ZAlignedSpheres(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere1 = Sphere(position=Vector(0, 0, 2), order=2)
        sphere2 = Sphere(position=Vector(0, 0, 5), order=2)
        sphere3 = Sphere(position=Vector(0, 0, 9), order=2)
        sphere4 = Sphere(position=Vector(0, 0, 20), order=2)
        self._solids.extend([sphere1, sphere2, sphere3, sphere4])


class DiagonallyAlignedSpheres(Scene):
    def __init__(self):
        super().__init__()
        self._create()

    def _create(self):
        sphere1 = Sphere(position=Vector(0, 0, 0), order=2)
        sphere2 = Sphere(position=Vector(3, 3, 3), order=2)
        sphere3 = Sphere(position=Vector(7, 7, 7), order=2)
        sphere4 = Sphere(position=Vector(20, 20, 20), order=2)
        self._solids.extend([sphere1, sphere2, sphere3, sphere4])
