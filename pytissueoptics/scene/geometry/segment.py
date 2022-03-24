from dataclasses import dataclass

from pytissueoptics.scene.geometry import Vector


@dataclass
class Segment:
    start: Vector
    end: Vector
