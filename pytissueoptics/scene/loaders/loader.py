import pathlib
from .parsers import Parser, OBJParser
from ..solids.solid import Solid
from ..geometry import Vector, Polygon


class Loader:
    """
    Base class to manage the conversion between files and Scene() or Solid() from
    various types of files.
    """
    def __init__(self):
        self._filepath: str = ""
        self._fileExtension: str = ""
        self._parser = None

    def _getFileExtension(self) -> str:
        return pathlib.Path(self._filepath).suffix

    def _selectParser(self):
        ext = self._fileExtension
        if ext == ".obj":
            self._parser = OBJParser(self._filepath)

        elif ext == ".dae":
            raise NotImplementedError

        elif ext == ".zmx":
            raise NotImplementedError

        else:
            raise ValueError("This format is not supported.")

    def _convert(self):
        if len(self._parser.objects) == 1:
            vertices = []
            surfacesGroups = {}
            for vertex in self._parser.vertices:
                vertices.append(Vector(*vertex))
            for objectName in self._parser.objects:
                for group in self._parser.objects[objectName]["Groups"]:
                    surfacesGroups[group] = []
                    for polygonIndices in self._parser.objects[objectName]["Groups"][group]["Polygons"]:
                        if len(polygonIndices) == 3:
                            surfacesGroups[group].append(Polygon(vertices=[vertices[polygonIndices[0]], vertices[polygonIndices[1]], vertices[polygonIndices[2]]]))
                        if len(polygonIndices) == 4:
                            surfacesGroups[group].append(Polygon(vertices=[vertices[polygonIndices[0]], vertices[polygonIndices[1]], vertices[polygonIndices[2]], vertices[polygonIndices[3]]]))
            solid = Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=surfacesGroups)
            return solid

        else:
            print("argh.")

    def load(self, filepath):
        self._filepath = filepath
        self._fileExtension = self._getFileExtension()
        self._selectParser()
        return self._convert()
