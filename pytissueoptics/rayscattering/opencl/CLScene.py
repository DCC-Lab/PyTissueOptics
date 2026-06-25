from typing import List, Optional

import numpy as np

from pytissueoptics.rayscattering.opencl.buffers import (
    LeafPolygonsCL,
    SolidCLInfo,
    SurfaceCLInfo,
    TreeNodeCL,
    TriangleCLInfo,
    flattenSpacePartition,
)
from pytissueoptics.rayscattering.opencl.buffers.materialCL import MaterialCL
from pytissueoptics.rayscattering.opencl.buffers.solidCandidateCL import SolidCandidateCL
from pytissueoptics.rayscattering.opencl.buffers.solidCL import SolidCL
from pytissueoptics.rayscattering.opencl.buffers.surfaceCL import SurfaceCL
from pytissueoptics.rayscattering.opencl.buffers.triangleCL import TriangleCL
from pytissueoptics.rayscattering.opencl.buffers.vertexCL import VertexCL
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.scene.tree import SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitThreeAxesConstructor

NO_LOG_ID = 0
WORLD_SOLID_ID = -1
NO_SURFACE_ID = -1
FIRST_SOLID_ID = 1
WORLD_SOLID_LABEL = "world"

# Default tree shape, mirroring the CPU FastIntersectionFinder defaults.
BVH_MAX_DEPTH = 20
BVH_MIN_LEAF_SIZE = 6

# Penalty multiplier applied to BVH traversal cost vs. the flat-list inner loop. The flat path
# is a tight 2-loop over contiguous arrays; BVH adds branchy memory access plus a private
# traversal stack, both of which hurt GPU occupancy. K=5 means we only switch to BVH once the
# flat-list worst case is at least ~5x the BVH worst case. Tunable from benchmarks.
BVH_KERNEL_PENALTY = 5


class CLScene:
    def __init__(self, scene: ScatteringScene, nWorkUnits: int, useBVH: Optional[bool] = None):
        self._sceneMaterials = scene.getMaterials()
        self._solidLabels = [solid.getLabel() for solid in scene.getSolids()]
        self._surfaceLabels = {}

        self._solidsInfo = []
        self._surfacesInfo = []
        self._trianglesInfo = []
        self._polygonToTriangleID: dict = {}
        self._vertices = []
        for solid in scene.solids:
            self._processSolid(solid)

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.solidCandidates = SolidCandidateCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(self._solidsInfo)
        self.surfaces = SurfaceCL(self._surfacesInfo)
        self.triangles = TriangleCL(self._trianglesInfo)
        self.vertices = VertexCL(self._vertices)

        self.useBVH, self.treeNodes, self.leafPolygons, self.nNodes = self._buildBVH(scene, useBVH)

    def _buildBVH(self, scene: ScatteringScene, useBVH: Optional[bool]):
        if useBVH is None:
            useBVH = self._bvhWorthwhile()
        if not useBVH or len(self._trianglesInfo) == 0:
            return False, TreeNodeCL([]), LeafPolygonsCL([]), np.uint32(0)

        partition = SpacePartition(
            scene.getBoundingBox(),
            scene.getPolygons(),
            NoSplitThreeAxesConstructor(),
            maxDepth=BVH_MAX_DEPTH,
            minLeafSize=BVH_MIN_LEAF_SIZE,
        )
        treeNodes, leafPolygons = flattenSpacePartition(partition, self._polygonToTriangleID)
        nNodes = np.uint32(len(treeNodes._nodes))
        return True, treeNodes, leafPolygons, nNodes

    def _bvhWorthwhile(self) -> bool:
        """
        Decide whether BVH traversal is likely to beat the flat-list path on the GPU.

        The flat path is dominated by an O(N_solids^2) bubble sort over solid bbox candidates
        plus, in the worst case, a linear scan over every polygon. The BVH path is roughly
        log2(N_polys) node tests plus one leaf-sized polygon scan, but each operation is more
        expensive on the GPU because of branchy memory access and a private traversal stack
        (charged via BVH_KERNEL_PENALTY).
        """
        nSolids = len(self._solidsInfo)
        nPolys = len(self._trianglesInfo)
        if nPolys == 0:
            return False
        flatCost = nSolids * nSolids + nPolys
        bvhCost = BVH_KERNEL_PENALTY * (int(np.log2(max(nPolys, 1))) + BVH_MIN_LEAF_SIZE)
        return flatCost > bvhCost

    def getMaterialID(self, material):
        if material is None:
            # Detector case. Set dummy value (not used).
            return 0
        return self._sceneMaterials.index(material)

    def getSolidID(self, solid):
        if solid is None:
            return WORLD_SOLID_ID
        return self._solidLabels.index(solid.getLabel()) + FIRST_SOLID_ID

    def getSolidLabel(self, solidID):
        if solidID == WORLD_SOLID_ID:
            return WORLD_SOLID_LABEL
        return self._solidLabels[solidID - FIRST_SOLID_ID]

    def getSolidIDs(self) -> List[int]:
        solidIDs = list(self._surfaceLabels.keys())
        solidIDs.insert(0, WORLD_SOLID_ID)
        return solidIDs

    def getSurfaceIDs(self, solidID):
        if solidID == WORLD_SOLID_ID:
            return [NO_SURFACE_ID]
        surfaceIDs = list(self._surfaceLabels[solidID].keys())
        surfaceIDs.insert(0, NO_SURFACE_ID)
        return surfaceIDs

    def getSurfaceLabel(self, solidID, surfaceID):
        if solidID == WORLD_SOLID_ID:
            return None
        if surfaceID == NO_SURFACE_ID:
            return None
        return self._surfaceLabels[solidID][surfaceID]

    def _processPolygon(self, polygon, surfaceLabel, surfaceID: int):
        outsideSolid = polygon.outsideEnvironment.solid
        if outsideSolid is not None and outsideSolid.getLabel() not in self._solidLabels:
            self._solidLabels.append(outsideSolid.getLabel())

        insideSolidLabel = polygon.insideEnvironment.solid.getLabel()
        if insideSolidLabel not in self._solidLabels:
            self._solidLabels.append(insideSolidLabel)

        solidID = self.getSolidID(polygon.insideEnvironment.solid)
        if solidID not in self._surfaceLabels:
            self._surfaceLabels[solidID] = {}

        if surfaceID not in self._surfaceLabels[solidID]:
            self._surfaceLabels[solidID][surfaceID] = surfaceLabel
        if outsideSolid is not None:
            outsideSolidID = self.getSolidID(outsideSolid)
            if outsideSolidID not in self._surfaceLabels:
                self._surfaceLabels[outsideSolidID] = {}
            self._surfaceLabels[outsideSolidID][surfaceID] = surfaceLabel

    def _compileSurface(self, polygonRef, firstPolygonID, lastPolygonID):
        insideEnvironment = polygonRef.insideEnvironment
        outsideEnvironment = polygonRef.outsideEnvironment
        insideMaterialID = self.getMaterialID(insideEnvironment.material)
        outsideMaterialID = self.getMaterialID(outsideEnvironment.material)
        insideSolidID = self.getSolidID(insideEnvironment.solid)
        outsideSolidID = self.getSolidID(outsideEnvironment.solid)
        toSmooth = polygonRef.toSmooth
        isDetector = insideEnvironment.solid.isDetector if insideEnvironment.solid else False
        detectorCosine = insideEnvironment.solid.detectorAcceptanceCosine if isDetector else 0.0

        self._surfacesInfo.append(
            SurfaceCLInfo(
                firstPolygonID,
                lastPolygonID,
                insideMaterialID,
                outsideMaterialID,
                insideSolidID,
                outsideSolidID,
                toSmooth,
                isDetector,
                detectorCosine,
            )
        )

    def _processSolid(self, solid):
        solidVertices = solid.getVertices()
        vertexToID = {id(v): i + len(self._vertices) for i, v in enumerate(solidVertices)}

        firstSurfaceID = len(self._surfacesInfo)
        for surfaceLabel in solid.surfaceLabels:
            surfacePolygons = solid.getPolygons(surfaceLabel)
            self._processSurface(surfaceLabel, surfacePolygons, vertexToID)

        lastSurfaceID = len(self._surfacesInfo) - 1
        self._vertices.extend(solidVertices)
        self._solidsInfo.append(SolidCLInfo(solid.bbox, firstSurfaceID, lastSurfaceID))

    def _processSurface(self, surfaceLabel, polygons, vertexToID):
        firstPolygonID = len(self._trianglesInfo)

        lastSolid = None
        for i, triangle in enumerate(polygons):
            # todo: consider skipping this step if the solid is not a stack.
            currentSolid = triangle.insideEnvironment.solid
            if lastSolid and lastSolid != currentSolid:
                self._compileSurface(
                    polygonRef=polygons[i - 1],
                    firstPolygonID=firstPolygonID,
                    lastPolygonID=len(self._trianglesInfo) - 1,
                )
                firstPolygonID = len(self._trianglesInfo)

            vertexIDs = [vertexToID[id(v)] for v in triangle.vertices]
            newSurfaceID = len(self._surfacesInfo)
            triangleID = len(self._trianglesInfo)
            self._trianglesInfo.append(TriangleCLInfo(vertexIDs, triangle.normal, newSurfaceID))
            self._polygonToTriangleID[id(triangle)] = triangleID
            self._processPolygon(triangle, surfaceLabel, surfaceID=newSurfaceID)
            lastSolid = currentSolid

        self._compileSurface(
            polygonRef=polygons[-1], firstPolygonID=firstPolygonID, lastPolygonID=len(self._trianglesInfo) - 1
        )
