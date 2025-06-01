import json
import os
import pickle
from typing import List, Optional, TextIO, Union

import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.display.views.view2D import View2D, ViewGroup
from pytissueoptics.rayscattering.display.views.viewFactory import ViewFactory
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger.listArrayContainer import ListArrayContainer
from pytissueoptics.scene.logger.logger import DataType, InteractionData, InteractionKey, Logger

from ..opencl.CLScene import NO_SOLID_LABEL
from .energyType import EnergyType


class EnergyLogger(Logger):
    _data: dict[InteractionKey, InteractionData]

    def __init__(
        self,
        scene: ScatteringScene,
        filepath: str = None,
        keep3D: bool = True,
        views: Union[ViewGroup, List[View2D]] = ViewGroup.ALL,
        defaultViewEnergyType: EnergyType = EnergyType.DEPOSITION,
        defaultBinSize: Union[float, tuple] = 0.01,
        infiniteLimits=((-5, 5), (-5, 5), (-5, 5)),
    ):
        """
        Log the energy deposited by scattering photons as well as the energy that crossed surfaces. Every interaction
        is linked to a specific solid and surface of the scene when applicable. This `EnergyLogger` has to be given to
        a `Source` instance when propagating.

        :param scene: The scene is required to auto-initialize the 2D views correctly. Even if `keep3D` is True, the
                scene may be used later to correctly create and display the 2D views.
        :param filepath: (Optional) Load the logger from the given file. When saving, the file will be overwritten.
        :param keep3D: (Default to True) If False, logged datapoints are automatically binned to predefined 2D `views`
                and the 3D data is discarded. This allows for a lightweight logger alternative. If True, the 3D data is
                kept and the 2D views are only computed later if displayed.
        :param views: The 2D views to track, particularly used when `keep3D` is set to False. Can be a ViewGroup flag
                or a list of View2D. Default to ViewGroup.ALL which includes the 3 default XYZ projections for the
                whole scene as well as for each solid, and a projection of each surface in the direction of the surface
                normal for both the energy entering and leaving the surface. When `keep3D` is False, ViewGroup.All is
                required to allow calculation of the detailed report for absorbance and transmittance. When `keep3D` is
                True, the 2D views data is only computed when displayed, so it is equivalent to setting `views` to None.
        :param defaultBinSize: The default bin size to use when binning the 3D data to 2D views. In the same physical
                units as the scene. Custom bin sizes can be specified in each View2D.
        :param infiniteLimits: The default limits to use for the 2D views when the scene is infinite (has no solids).
        """
        self._scene = scene
        self._keep3D = keep3D
        self._defaultBinSize = defaultBinSize
        self._infiniteLimits = infiniteLimits
        self._viewFactory = ViewFactory(scene, defaultBinSize, infiniteLimits, energyType=defaultViewEnergyType)

        self._sceneHash = hash(scene)
        self._defaultViews = views
        self._views = self._viewFactory.build(views)
        self._outdatedViews = set()
        self._nDataPointsRemoved = 0

        super().__init__(fromFilepath=filepath)

    def addView(self, view: View2D) -> bool:
        self._viewFactory.build([view])

        if self._viewExists(view):
            return True

        if self.isEmpty:
            self._views.append(view)
            return True

        if self.has3D:
            self._compileViews([view])
            self._views.append(view)
            return True

        for i, existingView in enumerate(self._views):
            if view.isContainedBy(existingView):
                view.initDataFrom(existingView)
                if existingView in self._outdatedViews:
                    self._outdatedViews.add(view)
                self._views.append(view)
                return True

        utils.warn(
            f"ERROR: Cannot create view {view.name}. The 3D data was discarded and the required data was not "
            f"found in existing views."
        )
        return False

    def updateView(self, view: View2D):
        if view in self._outdatedViews:
            self._compileViews([view])

    def showView(self, view: View2D = None, viewIndex: int = None, logScale: bool = True, colormap: str = "viridis"):
        assert viewIndex is not None or view is not None, "Either `viewIndex` or `view` must be specified."

        if viewIndex is None:
            created = self.addView(view)
            if not created:
                utils.warn(f"ERROR: Cannot display view {view.name}. Failed to create the view.")
                return
            viewIndex = self._getViewIndex(view)

        view = self.getView(viewIndex)
        self.updateView(view)

        view.show(logScale=logScale, colormap=colormap)

    def listViews(self):
        print("Available views:")
        for i, view in enumerate(self._views):
            print(f"\t{i}: {view.description}")

    def save(self, filepath: str = None):
        if filepath is None and self._filepath is None:
            filepath = self.DEFAULT_LOGGER_PATH
            utils.warn(f"No filepath specified. Saving to {filepath}.")
        elif filepath is None:
            filepath = self._filepath

        with open(filepath, "wb") as file:
            pickle.dump(
                (
                    self._data,
                    self.info,
                    self._labels,
                    self._views,
                    self._defaultViews,
                    self._outdatedViews,
                    self._nDataPointsRemoved,
                    self._sceneHash,
                    self.has3D,
                ),
                file,
            )

    def load(self, filepath: str):
        self._filepath = filepath

        if not os.path.exists(filepath):
            utils.warn(
                "No logger file found at '{}'. No data loaded, but it will create a new file "
                "at this location if the logger is saved later on.".format(filepath)
            )
            return

        with open(filepath, "rb") as file:
            (
                self._data,
                self.info,
                self._labels,
                self._views,
                oldDefaultViews,
                self._outdatedViews,
                self._nDataPointsRemoved,
                oldSceneHash,
                oldHas3D,
            ) = pickle.load(file)

        if oldSceneHash != self._sceneHash:
            utils.warn(
                "WARNING: The scene used to create the logger at '{}' is different from the current "
                "scene. This may corrupt statistics and visualization. Proceed at your own risk.".format(filepath)
            )
        if oldHas3D and not self._keep3D:
            utils.warn(
                "WARNING: The logger at '{}' use to store 3D data, but it was reloaded with keep3D=False. "
                "The 3D data will be compiled to 2D views and discarded.".format(filepath)
            )
        if not oldHas3D and self._keep3D:
            utils.warn(
                "WARNING: The logger at '{}' use to discard 3D data, but it was reloaded with keep3D=True. "
                "This may corrupt the statistics and the 3D visualization. Proceed at your own risk.".format(filepath)
            )
        if self._defaultViews != oldDefaultViews:
            utils.warn(
                "WARNING: Cannot provide new default views to a loaded logger from '{}'."
                "Using only the views from the file.".format(filepath)
            )

    @property
    def views(self) -> List[View2D]:
        return self._views

    def getView(self, index: int) -> View2D:
        if index < 0 or index >= len(self._views):
            raise IndexError(
                f"View index {index} is out of range [0, {len(self._views)}]. Use `.listViews()` to see "
                f"available views."
            )
        return self._views[index]

    def _getViewIndex(self, view: View2D) -> int:
        for i, v in enumerate(self._views):
            if v.isEqualTo(view):
                return i
        raise ValueError(f"View {view.name} not found in the list of views.")

    def _viewExists(self, view: View2D) -> bool:
        return any([view.isEqualTo(v) for v in self._views])

    @property
    def has3D(self) -> bool:
        return self._keep3D

    @property
    def defaultBinSize(self) -> float:
        return self._defaultBinSize

    @property
    def infiniteLimits(self) -> tuple:
        return self._infiniteLimits

    def getSolidLimits(self, solidLabel: str) -> List[List[float]]:
        return self._scene.getSolid(solidLabel).getBoundingBox().xyzLimits

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        """
        Used internally by `Source` when propagating photons. Overwrites the `Logger` method to automatically bin the
        data to 2D views if 3D data is being discarded.
        """
        super().logDataPointArray(array, key)
        self._outdatedViews = set(self._views)

        if not self._keep3D:
            self._compileViews(self._views)
            self._delete3DData()

    def logDataPoint(self, value: float, position: Vector, key: InteractionKey):
        self.logDataPointArray(np.array([[value, *position.array]]), key)

    def _compileViews(self, views: List[View2D]):
        for key, data in self._data.items():
            datapointsContainer: Optional[ListArrayContainer] = data.dataPoints
            if datapointsContainer is None or len(datapointsContainer) == 0:
                continue
            for view in views:
                if view.solidLabel and not utils.labelsEqual(view.solidLabel, key.solidLabel):
                    continue
                if view.surfaceLabel and not utils.labelsEqual(view.surfaceLabel, key.surfaceLabel):
                    continue
                if view.surfaceLabel is None and key.surfaceLabel is not None:
                    continue

                data = datapointsContainer.getData()
                if view.energyType == EnergyType.FLUENCE_RATE:
                    data = self._fluenceTransform(key, data)

                view.extractData(data)
        for view in views:
            self._outdatedViews.discard(view)

    def _delete3DData(self):
        self._nDataPointsRemoved += super().nDataPoints
        self._data.clear()

    @property
    def nDataPoints(self) -> int:
        """
        Overwrites the `Logger` method to return the total number of data points logged, including the ones discarded.
        """
        if self._keep3D:
            return super().nDataPoints
        else:
            return self._nDataPointsRemoved

    @property
    def isEmpty(self) -> bool:
        return self.nDataPoints == 0

    def logPoint(self, point: Vector, key: InteractionKey = None):
        raise NotImplementedError("Can only log data points to an EnergyLogger.")

    def logPointArray(self, array: np.ndarray, key: InteractionKey = None):
        raise NotImplementedError("Can only log data points to an EnergyLogger.")

    def logSegment(self, start: Vector, end: Vector, key: InteractionKey = None):
        raise NotImplementedError("Can only log data points to an EnergyLogger.")

    def logSegmentArray(self, array: np.ndarray, key: InteractionKey = None):
        raise NotImplementedError("Can only log data points to an EnergyLogger.")

    def getDataPoints(self, key: InteractionKey, energyType=EnergyType.DEPOSITION) -> np.ndarray:
        """All 3D data points recorded for this InteractionKey (not binned). Array of shape (n, 4) where
        the second axis is (value, x, y, z). The value can be the energy deposited, the fluence rate, or the
        energy that crossed the surface.

        :param key: Filtering the data by solidLabel and surfaceLabel.
        :param energyType: The type of volumetric energy to return when no surfaceLabel is given.

        :return: The data points (value, x, y, z) for the given solidLabel and surfaceLabel. If a surfaceLabel is given,
        the value corresponds to the energy that crossed the surface (positive when in the direction of the normal). If
        only a solidLabel is given, the value corresponds to the volumetric EnergyType at that point.
        """
        if energyType == EnergyType.FLUENCE_RATE:
            return self._getData(DataType.DATA_POINT, key, transform=self._fluenceTransform)

        return self._getData(DataType.DATA_POINT, key)

    def _fluenceTransform(self, key: InteractionKey, data: Optional[np.ndarray]) -> Optional[np.ndarray]:
        # Converts volumetric data to fluence rate when needed.
        if not key.volumetric or data is None:
            return data

        data[:, 0] = data[:, 0] / self._scene.getMaterial(key.solidLabel).mu_a
        return data

    def export(self, exportPath: str):
        """
        Export the raw 3D data points to a CSV file, along with the scene information to a JSON file.

        The data file <exportPath>.csv will be comma-delimited and will contain the following columns:
        - energy, x, y, z, solid_index, surface_index

        Two types of interactions are logged: scattering and surface crossings. In the first case, the energy will be
        the delta energy deposited at the point and the surface index will be -1. In the second case, the energy
        will be the total photon energy when crossing the surface, either as positive if leaving the surface
        (along the normal) or as negative if entering the surface.

        The scene information will be saved in a JSON file named <exportPath>.json, which includes details for each solid
        index and surface index, such as their labels, materials, and geometry. The world information is also exported
        as solid index -1.
        """
        if not self.has3D:
            utils.warn("Cannot export data when keep3D is False. No 3D data available.")
            return

        solidLabels = []
        for solid in self._scene.solids:
            if solid.isStack():
                solidLabels.extend(solid.getLayerLabels())
            else:
                solidLabels.append(solid.getLabel())
        solidLabels.sort()

        print("Exporting raw data to file...")
        filepath = f"{exportPath}.csv"
        with open(filepath, "w") as file:
            file.write("energy,x,y,z,solid_index,surface_index\n")
            self._writeKeyData(file, InteractionKey(NO_SOLID_LABEL), -1, -1)
            for i, solidLabel in enumerate(solidLabels):
                self._writeKeyData(file, InteractionKey(solidLabel), i, -1)
                for j, surfaceLabel in enumerate(self._scene.getSurfaceLabels(solidLabel)):
                    self._writeKeyData(file, InteractionKey(solidLabel, surfaceLabel), i, j)
        print(f"Exported data points to {filepath}")

        sceneInfo = {}
        material = self._scene.getWorldEnvironment().material
        sceneInfo["-1"] = {"label": "world", "material": material.__dict__ if material else None}
        for i, solidLabel in enumerate(solidLabels):
            material = self._scene.getMaterial(solidLabel)
            solid = self._scene.getSolid(solidLabel)
            surfaces = {}
            for j, surfaceLabel in enumerate(solid.surfaceLabels):
                normals = [s.normal for s in solid.getPolygons(surfaceLabel)[:2]]
                if len(normals) == 1 or normals[0] == normals[1]:
                    normal = normals[0].array
                else:
                    normal = None
                surfaces[j] = {"label": surfaceLabel, "normal": normal}

            sceneInfo[str(i)] = {
                "label": solidLabel,
                "type": solid.__class__.__name__,
                "material": material.__dict__ if material else None,
                "geometry": solid.geometryExport(),
                "surfaces": surfaces,
            }

        sceneFilepath = f"{exportPath}.json"
        with open(sceneFilepath, "w") as file:
            json.dump(sceneInfo, file, indent=4)
        print(f"Exported scene information to {sceneFilepath}")

    def _writeKeyData(self, file: TextIO, key: InteractionKey, solidIndex: int, surfaceIndex: int):
        if key not in self._data or self._data[key].dataPoints is None:
            return
        dataArray = self._data[key].dataPoints.getData().astype(str)
        dataArray = np.hstack((dataArray, np.full((dataArray.shape[0], 2), str(solidIndex))))
        dataArray[:, 5] = str(surfaceIndex)
        file.write("\n".join([",".join(row) for row in dataArray]) + "\n")
