import pathlib
from typing import List

from pytissueoptics.scene.loader.parsers import OBJParser
from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Vector, Triangle, SurfaceCollection


class Loader:
    """
    Base class to manage the conversion between files and Scene() or Solid() from
    various types of files.
    """
    def __init__(self):
        self._filepath: str = ""
        self._fileExtension: str = ""
        self._parser = None

    def load(self, filepath: str) -> List[Solid]:
        self._filepath = filepath
        self._fileExtension = self._getFileExtension()
        self._selectParser()
        return self._convert()

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

    def _convert(self) -> List[Solid]:
        vertices = []
        for vertex in self._parser.vertices:
            vertices.append(Vector(*vertex))

        solids = []
        for objectName, _object in self._parser.objects.items():
            surfaces = SurfaceCollection()
            for surfaceName, surface in _object.surfaces.items():
                surfaces.add(surfaceName, self._convertSurfaceToPolygons(surface, vertices))
            solids.append(Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=surfaces))

        return solids

    def _convertSurfaceToPolygons(self, surface: ParsedSurface, vertices: List[Vector]) -> List[Triangle]:
        surfaces = []
        for polygonIndices in surface.polygons:
            if len(polygonIndices) == 3:
                ai, bi, ci = polygonIndices
                surfaces.append(Triangle(vertices[ai], vertices[bi], vertices[ci]))

            elif len(polygonIndices) > 3:
                trianglesIndices = self._splitPolygonIndices(polygonIndices)
                for triangleIndex in trianglesIndices:
                    ai, bi, ci = triangleIndex
                    surfaces.append(Triangle(vertices[ai], vertices[bi], vertices[ci]))
        return surfaces

    @staticmethod
    def _splitPolygonIndices(polygonIndices: List[int]) -> List[List[int]]:
        trianglesIndices = []
        for i in range(len(polygonIndices)-2):
            trianglesIndices.append([polygonIndices[0], polygonIndices[i+1], polygonIndices[i+2]])
        return trianglesIndices
