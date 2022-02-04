
class Parser:
    """
    Base class to parse a file and extract relevant information.
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._objects = {}
        self._vertices = []
        self._normals = []
        self._texCoords = []
        self._polygons = []
        self._polyGroups = []
        self._currentObjectKey = -1
        self._currentGroupKey = -1
        self._material = None

    def _parse(self):
        raise NotImplementedError

    def _resetGroupIndex(self):
        self._currentGroupIndex = -1
