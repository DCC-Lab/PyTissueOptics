from pytissueoptics.scene.geometry import Vector, Triangle
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid


class Sphere(Solid):
    """
        The Sphere is the 3D analog to the circle. Meshing a sphere requires an infinite number of vertices.
        The position refers to the vector from global origin to its centroid.
        The radius of the sphere will determine the outermost distance from its centroid.

        This class offers two possible methods to generate the sphere mesh.
        - With Quads: Specify the number of separation lines on the vertical axis and the horizontal axis of the sphere.
        - With Triangle: Specify the order of splitting. This will generate what is known as an IcoSphere.
    """

    def __init__(self, radius: float = 1.0, order: int = 3,
                 position: Vector = Vector(0, 0, 0), material: Material = None,
                 primitive: str = primitives.DEFAULT):

        self._radius = radius
        self._order = order

        super().__init__(vertices=[], position=position, material=material, primitive=primitive)

    @property
    def radius(self):
        return self._radius

    def _computeTriangleMesh(self):
        """
        The most precise sphere approximation is the IcoSphere, which is generated from the platonic solid,
        the Icosahedron. It is built with 20 equilateral triangles with exactly the same angle between each.
        From Euler's method to generate the vertex for the icosahedron, we cross 3 perpendicular planes,
        with lenght=2 and width=2*phi. Joining adjscent vertices will produce the Icosahedron.

        From the Icosahedron, we can split each face in 4 triangles, in a recursive manner, to obtain an IcoSphere.
        The method goes as follow:
        1 - Find each mid-point between two connecting vertices on a triangle
        2 - Normalize those new points to project them onto the unit sphere.
        3 - Connect the new vertices in a way to make 4 new triangles.
        4 - Do these steps for each triangle (This will lead to redundant calculation, I am aware)
        5 - Replace the old surfaces by the new surfaces
        """

        self._computeFirstOrderTriangleMesh()

        for i in range(0, self._order):
            self._computeNextOrderTriangleMesh()

        self._setVerticesPositionsFromCenter()

    def _computeFirstOrderTriangleMesh(self):
        phi = (1.0 + 5.0 ** (1 / 2)) / 2.0
        xyPlaneVertices = [Vector(-1, phi, 0), Vector(1, phi, 0), Vector(-1, -phi, 0), Vector(1, -phi, 0)]
        yzPlaneVertices = [Vector(0, -1, phi), Vector(0, 1, phi), Vector(0, -1, -phi), Vector(0, 1, -phi)]
        xzPlaneVertices = [Vector(phi, 0, -1), Vector(phi, 0, 1), Vector(-phi, 0, -1), Vector(-phi, 0, 1)]
        self._vertices = [*xyPlaneVertices, *yzPlaneVertices, *xzPlaneVertices]
        V = self._vertices

        self._surfaces.add("Sphere", [Triangle(V[0], V[11], V[5]), Triangle(V[0], V[5], V[1]),
                                      Triangle(V[0], V[1], V[7]), Triangle(V[0], V[7], V[10]),
                                      Triangle(V[0], V[10], V[11]), Triangle(V[1], V[5], V[9]),
                                      Triangle(V[5], V[11], V[4]), Triangle(V[11], V[10], V[2]),
                                      Triangle(V[10], V[7], V[6]), Triangle(V[7], V[1], V[8]),
                                      Triangle(V[3], V[9], V[4]), Triangle(V[3], V[4], V[2]),
                                      Triangle(V[3], V[2], V[6]), Triangle(V[3], V[6], V[8]),
                                      Triangle(V[3], V[8], V[9]), Triangle(V[4], V[9], V[5]),
                                      Triangle(V[2], V[4], V[11]), Triangle(V[6], V[2], V[10]),
                                      Triangle(V[8], V[6], V[7]), Triangle(V[9], V[8], V[1])])

    def _computeNextOrderTriangleMesh(self):
        newPolygons = []
        for j, polygon in enumerate(self.getPolygons()):
            ai = self._createMidVertex(polygon.vertices[0], polygon.vertices[1])
            bi = self._createMidVertex(polygon.vertices[1], polygon.vertices[2])
            ci = self._createMidVertex(polygon.vertices[2], polygon.vertices[0])

            self._vertices.extend([ai, bi, ci])

            newPolygons.append(Triangle(polygon.vertices[0], ai, ci))
            newPolygons.append(Triangle(polygon.vertices[1], bi, ai))
            newPolygons.append(Triangle(polygon.vertices[2], ci, bi))
            newPolygons.append(Triangle(ai, bi, ci))

        self._surfaces.setPolygons("Sphere", newPolygons)

    @staticmethod
    def _createMidVertex(p1, p2):
        middle = Vector((p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2)
        return middle

    def _setVerticesPositionsFromCenter(self):
        for vertex in self._vertices:
            vertex.normalize()
            vertex.multiply(self._radius)

    def _computeQuadMesh(self):
        raise NotImplementedError

    def contains(self, *vertices: Vector) -> bool:
        """ Only returns true if all vertices are inside the minimum radius of the sphere
        (more restrictive with low order spheres). """
        minRadius = self._getMinimumRadius()
        for vertex in vertices:
            relativeVertex = vertex - self.position
            if relativeVertex.getNorm() >= minRadius:
                return False
        return True

    def _getMinimumRadius(self) -> float:

    def _radiusTowards(self, vertex) -> float:
        return self.radius
        aPolygon = self.surfaces.getPolygons()[0]
        centerVertex = Vector(0, 0, 0)
        for vertex in aPolygon.vertices:
            centerVertex.add(vertex)
        centerVertex.divide(len(aPolygon.vertices))
        centerVertex.subtract(self.position)
        return centerVertex.getNorm()
