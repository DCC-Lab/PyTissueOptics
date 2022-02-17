from pytissueoptics.scene.tree import TreeStrategy


class QuadTreeStrategy(TreeStrategy):
    def _loadComponents(self):
        raise NotImplementedError
