from typing import List

from pytissueoptics.scene.geometry import Polygon, BoundingBox


def meanCentroid(axis: str, polygons: List[Polygon]):
    average = 0
    for polygon in polygons:
        if axis == "x":
            average += polygon.centroid.x
        elif axis == "y":
            average += polygon.centroid.y
        elif axis == "z":
            average += polygon.centroid.z
    average = average / len(polygons)
    return average

def getPolygonsBbox(polygons: List[Polygon]):
    bbox = BoundingBox([0, 0], [0, 0], [0, 0])
    for polygon in polygons:
        bbox.extendTo(polygon.bbox)
    return bbox
