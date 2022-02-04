
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
        self._currentObjectKey = ""
        self._currentGroupKey = ""
        self._checkFileExtension()
        self._parse()

    def _checkFileExtension(self):
        raise NotImplementedError

    def _parse(self):
        raise NotImplementedError
