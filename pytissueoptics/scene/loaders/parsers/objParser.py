from ..parsers import Parser


class OBJParser(Parser):

    def _checkFileExtension(self):
        if self._filepath.endswith('.obj'):
            return
        else:
            raise TypeError

    def _parse(self):
        for line in open(self._filepath, "r"):
            if line.startswith('#'):
                continue

            values = line.split()
            if not values:
                continue

            if values[0] == 'v':
                v = map(float, values[1:4])
                self._vertices.append(v)

            elif values[0] == 'vn':
                vn = map(float, values[1:4])
                self._normals.append(vn)

            elif values[0] == 'vt':
                vt = map(float, values[1:3])
                self._texCoords.append(vt)

            elif values[0] in ('usemtl', 'usemat'):
                self._objects[self._currentObjectKey]["Material"] = values[1]

            elif values[0] == 'f':
                faceIndices = []
                texCoordsIndices = []
                normalIndices = []
                for verticesIndices in values[1:]:
                    "faces indices have this format: 'v1/vt1/vn1 or v2/vt2 or v3//vn3'."
                    vertexIndices = verticesIndices.split('/')
                    faceIndices.append(int(vertexIndices[0]))
                    if len(vertexIndices) >= 2 and len(vertexIndices[1]) > 0:
                        texCoordsIndices.append(int(vertexIndices[1]))
                    else:
                        texCoordsIndices.append(0)
                    if len(vertexIndices) == 3 and len(vertexIndices[2]) > 0:
                        normalIndices.append(int(vertexIndices[2]))
                    else:
                        normalIndices.append(0)

                self._objects[self._currentObjectKey]["PolygonGroups"][self._currentGroupKey].append(faceIndices)

            elif values[0] == 'o':
                self._currentObjectKey = values[1]
                self._resetGroupKey()
                self._objects[self._currentObjectKey] = {}

            elif values[0] == 'g':
                self._currentGroupKey = values[1]
                self._objects[self._currentObjectKey]["PolygonGroups"][self._currentGroupKey] = []

    def _resetGroupKey(self):
        self._currentGroupKey = "noGroup"
