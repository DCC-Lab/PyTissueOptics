from typing import List
from pytissueoptics.scene.tree.splitUtils import AxisSelector


class QuadAxisSelector(AxisSelector):
    def _run(self) -> List[str]:
        raise NotImplementedError
