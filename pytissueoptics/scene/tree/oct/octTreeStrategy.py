from pytissueoptics.scene.tree import TreeStrategy


class OctTreeStrategy(TreeStrategy):
    def _loadComponents(self):
        raise NotImplementedError
