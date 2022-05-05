import pathlib
from typing import List

from pytissueoptics.scene.loader.parsers import OBJParser
from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Vector, Triangle, SurfaceCollection, Quad, Polygon, Vertex


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
            vertices.append(Vertex(*vertex))

        solids = []
        for objectName, _object in self._parser.objects.items():
            surfaces = SurfaceCollection()
            for surfaceLabel, surface in _object.surfaces.items():
                surfaces.add(surfaceLabel, self._convertSurfaceToPolygons(surface, vertices))
            solids.append(Solid(position=Vector(0, 0, 0), vertices=vertices, surfaces=surfaces))

        return solids

    @staticmethod
    def _convertSurfaceToPolygons(surface: ParsedSurface, vertices: List[Vertex]) -> List[Triangle]:
        polygons = []
        for polygonIndices in surface.polygons:
            polygonVertices = [vertices[i] for i in polygonIndices]
            if len(polygonIndices) == 3:
                polygon = Triangle(*polygonVertices)
            elif len(polygonIndices) == 4:
                polygon = Quad(*polygonVertices)
            else:
                polygon = Polygon(polygonVertices)
            polygons.append(polygon)
        return polygons
