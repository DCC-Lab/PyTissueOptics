from typing import List

from pytissueoptics.scene.loader.parsers.parsedObject import ParsedObject
from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface
from pytissueoptics.scene.loader.parsers.parser import Parser
from pytissueoptics.scene.utils.progressBar import progressBar


class OBJParser(Parser):
    def __init__(self, filepath: str, showProgress: bool = True):
        super().__init__(filepath, showProgress)

    def _checkFileExtension(self):
        if self._filepath.endswith('.obj'):
            return
        else:
            raise TypeError

    def _parse(self, showProgress: bool = True):
        """
        The .OBJ file format is well described here: https://en.wikipedia.org/wiki/Wavefront_.obj_file
        Summary of important points for our purposes
        - The indexing starts at '1' instead of '0'
        - Faces indices have this format: 'v1/vt1/vn1 or v2/vt2 or v3//vn3'.
        - Groups start with 'g'
        - New objects will start with 'o'
        """
        self._PARSE_MAP = {'v': self._parseVertices,
                           'vt': self._parseTexCoords,
                           'vn': self._parseNormals,
                           'usemtl': self._parseMaterial,
                           'usemat': self._parseMaterial,
                           'f': self._parseFace,
                           'g': self._parseGroup,
                           'o': self._parseObject}

        with open(self._filepath, "r") as file:
            lines = [line.strip('\n') for line in file.readlines() if line != "\n"]

        for i in progressBar(range(len(lines)), desc="Parsing File '{}'".format(self._filepath.split('/')[-1]),
                             unit=" lines", disable=not showProgress):
            self._parseLine(lines[i])

    def _parseLine(self, line: str):
        if line.startswith('#'):
            return

        values = line.split()
        if not values:
            return

        typeChar = values[0]
        if typeChar not in self._PARSE_MAP:
            return

        self._PARSE_MAP[typeChar](values)

    def _parseVertices(self, values: List[str]):
        v = list(map(float, values[1:4]))
        self._vertices.append(v)

    def _parseNormals(self, values: List[str]):
        vn = list(map(float, values[1:4]))
        self._normals.append(vn)

    def _parseTexCoords(self, values: List[str]):
        vt = list(map(float, values[1:3]))
        self._textureCoords.append(vt)

    def _parseMaterial(self, values: List[str]):
        self._objects[self._currentObjectName].material = values[1]

    def _parseFace(self, values: List[str]):
        faceIndices = []
        texCoordsIndices = []
        normalIndices = []

        for verticesIndices in values[1:]:
            vertexIndices = verticesIndices.split('/')
            faceIndices.append(int(vertexIndices[0]) - 1)

            if len(vertexIndices) >= 2 and len(vertexIndices[1]) > 0:
                texCoordsIndices.append(int(vertexIndices[1]) - 1)
            else:
                texCoordsIndices.append(0)
            if len(vertexIndices) == 3 and len(vertexIndices[2]) > 0:
                normalIndices.append(int(vertexIndices[2]) - 1)
            else:
                normalIndices.append(0)

        self._checkForNoObject()
        self._checkForNoSurface()

        self._objects[self._currentObjectName].surfaces[self._currentSurfaceLabel].polygons.append(faceIndices)
        self._objects[self._currentObjectName].surfaces[self._currentSurfaceLabel].normals.append(normalIndices)
        self._objects[self._currentObjectName].surfaces[self._currentSurfaceLabel].texCoords.append(texCoordsIndices)

    def _parseObject(self, values: List[str]):
        try:
            self._currentObjectName = values[1]
        except IndexError:
            self._currentObjectName = self.NO_OBJECT
        self._resetSurfaceLabel()
        self._objects[self._currentObjectName] = ParsedObject(material="", surfaces={})

    def _parseGroup(self, values: List[str]):
        try:
            self._currentSurfaceLabel = values[1]
        except IndexError:
            self._currentSurfaceLabel = self.NO_SURFACE
        self._checkForNoObject()
        self._validateSurfaceLabel()
        self._objects[self._currentObjectName].surfaces[self._currentSurfaceLabel] = ParsedSurface(polygons=[],
                                                                                                   normals=[],
                                                                                                   texCoords=[])

    def _checkForNoObject(self):
        if len(self._objects) == 0 and self._currentObjectName == self.NO_OBJECT:
            self._objects = {
                self.NO_OBJECT: ParsedObject(material="", surfaces={})}

    def _checkForNoSurface(self):
        if len(self._objects[self._currentObjectName].surfaces) == 0 and self._currentSurfaceLabel == self.NO_SURFACE:
            self._objects[self._currentObjectName].surfaces[self.NO_SURFACE] = ParsedSurface(polygons=[], normals=[],
                                                                                             texCoords=[])
