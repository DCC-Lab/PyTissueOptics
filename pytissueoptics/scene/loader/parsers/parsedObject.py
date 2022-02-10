from dataclasses import dataclass
from typing import Dict

from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface
from pytissueoptics.scene.materials import Material


@dataclass
class ParsedObject:
    material: Material
    surfaces: Dict[str, ParsedSurface]
