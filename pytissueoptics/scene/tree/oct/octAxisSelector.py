from typing import List
from pytissueoptics.scene.tree.splitUtils import AxisSelector


class OctAxisSelector(AxisSelector):
    def _run(self) -> List[str]:
        raise NotImplementedError
