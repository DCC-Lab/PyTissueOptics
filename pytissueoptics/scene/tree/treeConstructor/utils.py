from typing import List

from pytissueoptics.scene.geometry import Polygon


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
