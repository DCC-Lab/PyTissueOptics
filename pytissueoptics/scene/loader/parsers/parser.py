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
    {"noObject":{
        "Material":Material(),
        "Groups":{
            "NoGroup":{
                "Polygons":[], 
                "Normals":[],
                "TexCoords":[]}},
    "objectName":{
        "Material":Material(),
        "Groups":{
            "NoGroup":{
                "Polygons":[],
                "Normals":[],
                "TexCoords":[]}
            "Group0":{
                "Polygons":[],
                "Normals":[],
                "TexCoords":[]}}
                }
    }
    Other components, such as the vertices, dont need to be stored in the dictionary
    The reason is that it is a global entity that has no complexity and is always needed
    for the conversion later down the line.
    """
    NO_OBJECT = "noObject"
    NO_SURFACE = "noSurface"

    def __init__(self, filepath: str):
        self._filepath = filepath
        self._objects: Dict[str, ParsedObject] = {}
        self._vertices: List[List[float]] = []
        self._normals: List[List[float]] = []
        self._texCoords: List[List[float]] = []
        self._currentObjectName: str = self.NO_OBJECT
        self._currentSurfaceName: str = self.NO_SURFACE
        self._checkFileExtension()
        self._parse()

    def _checkFileExtension(self):
        raise NotImplementedError

    def _parse(self):
        raise NotImplementedError

    def _resetSurfaceName(self):
        self._currentSurfaceName = self.NO_SURFACE

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
