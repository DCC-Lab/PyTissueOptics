import pathlib
from typing import List

from pytissueoptics.scene.geometry import SurfaceCollection, Triangle, Vector, Vertex, primitives
from pytissueoptics.scene.loader.parsers import OBJParser
from pytissueoptics.scene.loader.parsers.parsedSurface import ParsedSurface
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.utils.progressBar import progressBar


class Loader:
    """
    Base class to manage the conversion between files and Scene() or Solid() from
    various types of files.
    """

    def __init__(self):
        self._filepath: str = ""
        self._fileExtension: str = ""
        self._parser = None

    def load(self, filepath: str, showProgress: bool = True) -> List[Solid]:
        self._filepath = filepath
        self._fileExtension = self._getFileExtension()
        self._selectParser(showProgress)
        return self._convert(showProgress)

    def _getFileExtension(self) -> str:
        return pathlib.Path(self._filepath).suffix

    def _selectParser(self, showProgress: bool = True):
        ext = self._fileExtension
        if ext == ".obj":
            self._parser = OBJParser(self._filepath, showProgress)
        else:
            raise NotImplementedError("This format is not supported.")

    def _convert(self, showProgress: bool = True) -> List[Solid]:
        vertices = []
        for vertex in self._parser.vertices:
            vertices.append(Vertex(*vertex))

        totalProgressBarLength = 0
        for objectName, _object in self._parser.objects.items():
            totalProgressBarLength += len(_object.surfaces.items())
        pbar = progressBar(
            total=totalProgressBarLength,
            desc="Converting File '{}'".format(self._filepath.split("/")[-1]),
            unit="surfaces",
            disable=not showProgress,
        )

        solids = []
        for objectName, _object in self._parser.objects.items():
            surfaces = SurfaceCollection()
            for surfaceLabel, surface in _object.surfaces.items():
                surfaces.add(surfaceLabel, self._convertSurfaceToTriangles(surface, vertices))
                pbar.update(1)
            solids.append(
                Solid(
                    position=Vector(0, 0, 0),
                    vertices=vertices,
                    surfaces=surfaces,
                    primitive=primitives.POLYGON,
                    label=objectName,
                )
            )

        pbar.close()
        return solids

    @staticmethod
    def _convertSurfaceToTriangles(surface: ParsedSurface, vertices: List[Vertex]) -> List[Triangle]:
        """Converting to triangles only since loaded polygons are often not planar."""
        triangles = []
        for polygonIndices in surface.polygons:
            polygonVertices = [vertices[i] for i in polygonIndices]
            for i in range(len(polygonVertices) - 2):
                triangles.append(Triangle(polygonVertices[0], polygonVertices[i + 1], polygonVertices[i + 2]))
        return triangles
