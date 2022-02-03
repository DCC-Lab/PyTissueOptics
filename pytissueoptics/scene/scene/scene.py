from typing import List

from pytissueoptics.scene.solids import Solid


class Scene:
    def __init__(self, solids: List[Solid]):
        self.solids = solids
