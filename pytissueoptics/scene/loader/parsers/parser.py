from typing import List, Dict

from pytissueoptics.scene.loader.parsers.parsedObject import ParsedObject


class Parser:
    """
    Parser base class to parse a file and extract relevant information into a structure.
    Each parser will manage one file type and convey the information into a standard format
    that will then be converted to the Scene language via the Loader class. The parser
    is the step in-between reading the file and converting it into the correct python objects
    and its dissociation from the Loader is mainly for clarity.

    The standard interface for the parser _object will look like this:
    self._object: Dict[str:, ParsedObject:]
    All the object will have their name in the dictionary and pointing to a parsedObject dataclass
    ParsedObject dataclass will contain a Material and surfaces:Dict[str, ParsedSurface]
    ParsedSurfaces will contain the polygon, normal and textureCoordinate indices.

    Other components, such as the vertices, dont need to be stored in the dictionary
    The reason is that it is a global entity that has no complexity and is always needed
    for the conversion later down the line.
    """
    NO_OBJECT = "noObject"
    NO_SURFACE = "noSurface"

    def __init__(self, filepath: str, showProgress: bool = True):
        self._filepath = filepath
        self._objects: Dict[str, ParsedObject] = {}
        self._vertices: List[List[float]] = []
        self._normals: List[List[float]] = []
        self._texCoords: List[List[float]] = []
        self._currentObjectName: str = self.NO_OBJECT
        self._currentSurfaceLabel: str = self.NO_SURFACE
        self._checkFileExtension()
        self._parse(showProgress)

    def _checkFileExtension(self):
        raise NotImplementedError

    def _parse(self, showProgress: bool = True):
        raise NotImplementedError

    def _resetSurfaceLabel(self):
        self._currentSurfaceLabel = self.NO_SURFACE

    def _validateSurfaceLabel(self):
        if self._currentSurfaceLabel not in self._objects[self._currentObjectName].surfaces:
            return
        idx = 0
        while f"{self._currentSurfaceLabel}_{idx}" in self._objects[self._currentObjectName].surfaces:
            idx += 1
        self._currentSurfaceLabel = f"{self._currentSurfaceLabel}_{idx}"

    @property
    def vertices(self):
        return self._vertices

    @property
    def normals(self):
        return self._normals

    @property
    def texCoords(self):
        return self._texCoords

    @property
    def objects(self):
        return self._objects
