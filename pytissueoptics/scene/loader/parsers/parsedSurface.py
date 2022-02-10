from dataclasses import dataclass
from typing import List


@dataclass
class ParsedSurface:
    polygons: List[List[int]]
    normals: List[List[int]]
    texCoords: List[List[int]]
