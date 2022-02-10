from typing import List, Dict


class Parser:
    """
    Parser base class to parse a file and extract relevant information into a structure.
    Each parser will manage one file type and convey the information into a standard format
    that will then be converted to the Scene language via the a Loader class. The parser
    is the step in-between Reading the file and converting it into the correct python objects
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
    Other components, such as the vertices, dont need to be stored in the dictionnary
    The reason is that it is a global entity that has no complexity and is always needed
    for the conversion later down the line.
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
