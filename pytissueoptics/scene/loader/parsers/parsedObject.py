from dataclasses import dataclass
from typing import Dict

from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface


@dataclass
class ParsedObject:
    material: str
    surfaces: Dict[str, ParsedSurface]
