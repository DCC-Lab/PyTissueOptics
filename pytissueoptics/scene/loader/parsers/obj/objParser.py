from typing import List, Any

from pytissueoptics.scene.loader.parsers import Parser
from pytissueoptics.scene.loader.parsers.parsedObject import ParsedObject
from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface


class OBJParser(Parser):
    def _checkFileExtension(self):
        if self._filepath.endswith('.obj'):
            return
        else:
            raise TypeError

    def _parse(self):
        """
        The .OBJ file format is well described here: https://en.wikipedia.org/wiki/Wavefront_.obj_file
        Summary of important points for our purposes
        - The indexing starts at '1' instead of '0'
        - Faces indices have this format: 'v1/vt1/vn1 or v2/vt2 or v3//vn3'.
        - Groups start with 'g'
        - New objects will start with 'o'
        """
        file = open(self._filepath, "r")
        for line in file:
            if line.startswith('#'):
                continue

            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                self._vertices.append(v)

            elif values[0] == 'vn':
                vn = list(map(float, values[1:4]))
                self._normals.append(vn)

            elif values[0] == 'vt':
                vt = list(map(float, values[1:3]))
                self._texCoords.append(vt)

            elif values[0] in ('usemtl', 'usemat'):
                self._objects[self._currentObjectName].material = values[1]

            elif values[0] == 'f':
                self._parseFace(values)

            elif values[0] == 'o':
                self._currentObjectName = values[1]
                self._resetSurfaceName()
                self._objects[self._currentObjectName] = ParsedObject(material="", surfaces={})

            elif values[0] == 'g':
                self._currentSurfaceName = values[1]
                self._checkForNoObject()
                self._objects[self._currentObjectName].surfaces[self._currentSurfaceName] = ParsedSurface(polygons=[],
                                                                                                          normals=[],
                                                                                                          texCoords=[])
        file.close()

    def _parseFace(self, values: List[Any]):
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

        self._objects[self._currentObjectName].surfaces[self._currentSurfaceName].polygons.append(faceIndices)
        self._objects[self._currentObjectName].surfaces[self._currentSurfaceName].normals.append(normalIndices)
        self._objects[self._currentObjectName].surfaces[self._currentSurfaceName].texCoords.append(texCoordsIndices)

    def _checkForNoObject(self):
        if len(self._objects) == 0 and self._currentObjectName == self.NO_OBJECT:
            self._objects = {
                self.NO_OBJECT: ParsedObject(material="", surfaces={})}

    def _checkForNoSurface(self):
        if len(self._objects[self._currentObjectName].surfaces) == 0 and self._currentSurfaceName == self.NO_SURFACE:
            self._objects[self._currentObjectName].surfaces[self.NO_SURFACE] = ParsedSurface(polygons=[], normals=[],
                                                                                             texCoords=[])

    def _resetSurfaceName(self):
        self._currentSurfaceName = self.NO_SURFACE
