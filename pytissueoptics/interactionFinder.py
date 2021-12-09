from typing import List

from pytissueoptics import Geometry


class InteractionFinder:
    def __init__(self, geometries: List[Geometry]):
        self.geometries = geometries


class SimpleInteractionFinder(InteractionFinder):
    pass
