import pathlib
from .parsers import OBJParser
from ..solids.solid import Solid
from ..geometry import Vector, Polygon


class Loader:
    """
    Base class to manage the conversion between files and Scene() or Solid() from
    various types of files.
    """
    def __init__(self, filepath: str):
        self._filepath = filepath
        self._fileExtension = self._getFileExtension()
        self._parser = self._selectParser()
        self._convert()

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

    def _convert(self):
        if len(self._parser.objects) == 1:
            vertices = []
            surfacesGroups = {}
            for vertex in self._parser.vertices:
                vertices.append(Vector(*vertex))
            for group in self._parser.objects[0]["Groups"]:
                surfacesGroups[group] = []
                for polygon in self._parser.objects[0]["Groups"][group]["Polygons"]:
                    surfacesGroups[group].append(Polygon)
        else:
            print("argh.")