from typing import List

from pytissueoptics.scene.geometry import Polygon, Vector, Vertex


def getSmoothNormal(polygon: Polygon, position: Vector) -> Vector:
    """ If the intersecting polygon was prepared for smoothing (i.e. it has vertex
    normals), we interpolate the normal at the intersection point using the normal
    of all its vertices. The interpolation is done using the general barycentric
    coordinates algorithm from http://www.geometry.caltech.edu/pubs/MHBD02.pdfv. """
    if not polygon.toSmooth:
        return polygon.normal

    weights = _getBarycentricWeights(polygon.vertices, position)

    smoothNormal = Vector(0, 0, 0)
    for weight, vertex in zip(weights, polygon.vertices):
        smoothNormal += vertex.normal * weight
    smoothNormal.normalize()

    return smoothNormal


def _getBarycentricWeights(vertices: List[Vertex], position: Vector) -> List[float]:
    weights = []
    n = len(vertices)
    for i, vertex in enumerate(vertices):
        prevVertex = vertices[(i - 1) % n]
        nextVertex = vertices[(i + 1) % n]
        w = (_cotangent(position, vertex, prevVertex) +
             _cotangent(position, vertex, nextVertex)) / (position - vertex).getNorm() ** 2
        weights.append(w)
    return [w / sum(weights) for w in weights]


def _cotangent(a: Vector, b: Vector, c: Vector) -> float:
    """ Cotangent of triangle abc at vertex b. """
    ba = a - b
    bc = c - b
    return bc.dot(ba) / bc.cross(ba).getNorm()
