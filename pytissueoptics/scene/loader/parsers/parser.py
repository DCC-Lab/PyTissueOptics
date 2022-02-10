from typing import List, Dict


class Parser:
    """
    Base class to parse a file and extract relevant information.
    """

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._objects: Dict = {}
        self._vertices: List[List] = []
        self._normals: List[List] = []
        self._texCoords: List[List] = []
        self._currentObjectKey: str = "noObject"
        self._currentGroupKey: str = "noGroup"
        self._checkFileExtension()
        self._parse()

    def _checkFileExtension(self):
        raise NotImplementedError

    def _parse(self):
        raise NotImplementedError

    def _resetGroupKey(self):
        self._currentGroupKey = "noGroup"

    @property
    def vertices(self) -> List[List]:
        return self._vertices

    @property
    def normals(self) -> List[List]:
        return self._normals

    @property
    def texCoords(self) -> List[List]:
        return self._texCoords

    @property
    def objects(self) -> Dict:
        return self._objects
