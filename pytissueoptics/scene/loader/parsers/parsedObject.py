from dataclasses import dataclass
from typing import List

from pytissueoptics.scene.loader.parsers import ParsedSurface
from pytissueoptics.scene.materials import Material


@dataclass
class ParsedObject:
    name: str
    material: Material
    groups: List[ParsedSurface]
