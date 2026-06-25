from typing import Dict, List, Tuple

import numpy as np

from pytissueoptics.scene.geometry import Polygon
from pytissueoptics.scene.tree import Node, SpacePartition

from .CLObject import CLObject, cl

# Sentinel used in TreeNode.offset to mark "no right child" on internal nodes.
# UINT32_MAX is safe: a real child offset can never equal it (we cap tree size well below 2^32).
NO_CHILD = np.uint32(0xFFFFFFFF)


class TreeNodeCL(CLObject):
    """
    DFS pre-order flattening of a scene SpacePartition. Each entry is either a leaf or an internal
    node:

      polygonCount > 0  -> leaf;   `offset` indexes into LeafPolygonsCL, range [offset, offset + polygonCount)
      polygonCount == 0 -> internal; left child is always at index+1; `offset` is the right child
                          index, or NO_CHILD (UINT32_MAX) if the node only has a left child.

    The kernel walks this representation iteratively with a small private-memory stack.
    """

    STRUCT_NAME = "TreeNode"
    STRUCT_DTYPE = np.dtype(
        [
            ("bbox_min", cl.cltypes.float3),
            ("bbox_max", cl.cltypes.float3),
            ("polygonCount", cl.cltypes.uint),
            ("offset", cl.cltypes.uint),
        ]
    )

    def __init__(self, nodes: List[Tuple[float, float, float, float, float, float, int, int]]):
        # nodes is a list of (xMin, yMin, zMin, xMax, yMax, zMax, polygonCount, offset).
        # Skip the auto-emitted C declaration: the TreeNode struct is hand-written in
        # intersection.c so it can be referenced by the Scene struct even on kernel paths
        # that do not pass a TreeNodeCL buffer. The dtype is still registered host-side via
        # match_dtype_to_c_struct so buffer reads/writes work normally.
        self._nodes = nodes
        super().__init__(skipDeclaration=True, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._nodes), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, n in enumerate(self._nodes):
            buffer[i]["bbox_min"][0] = np.float32(n[0])
            buffer[i]["bbox_min"][1] = np.float32(n[1])
            buffer[i]["bbox_min"][2] = np.float32(n[2])
            buffer[i]["bbox_max"][0] = np.float32(n[3])
            buffer[i]["bbox_max"][1] = np.float32(n[4])
            buffer[i]["bbox_max"][2] = np.float32(n[5])
            buffer[i]["polygonCount"] = np.uint32(n[6])
            buffer[i]["offset"] = np.uint32(n[7])
        return buffer


class LeafPolygonsCL(CLObject):
    """Flat array of triangle (polygon) IDs referenced by leaf nodes in TreeNodeCL."""

    STRUCT_NAME = None
    STRUCT_DTYPE = None

    def __init__(self, polygonIDs: List[int]):
        self._polygonIDs = polygonIDs
        super().__init__(buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._polygonIDs), 1)
        buffer = np.zeros(bufferSize, dtype=np.uint32)
        if self._polygonIDs:
            buffer[: len(self._polygonIDs)] = np.array(self._polygonIDs, dtype=np.uint32)
        return buffer

    @classmethod
    def getItemSize(cls) -> int:
        return np.dtype(np.uint32).itemsize


def flattenSpacePartition(
    partition: SpacePartition, polygonToTriangleID: Dict[int, int]
) -> Tuple[TreeNodeCL, LeafPolygonsCL]:
    """
    Walk the tree in DFS pre-order and emit the flat (TreeNodeCL, LeafPolygonsCL) pair.

    `polygonToTriangleID` maps id(Polygon) -> index into the TriangleCL buffer; this is built by
    CLScene as it processes solids, before the SpacePartition is constructed. Polygons that appear
    in more than one leaf (which the no-split constructors do for triangles straddling a split
    plane) are emitted once per leaf.
    """
    # An empty partition would be flattened into a single node with polygonCount == 0 and
    # offset == 0 - which the kernel parses as an internal node pointing to children that do
    # not exist (out-of-bounds read at index 1). Refuse to build such a tree; the caller
    # (CLScene._buildBVH) already short-circuits on empty scenes, so reaching this point is
    # always a misuse.
    if not partition.root.polygons:
        raise ValueError(
            "Cannot flatten a SpacePartition with no polygons. The kernel cannot distinguish "
            "an empty leaf from an internal node."
        )

    nodes: List[Tuple[float, float, float, float, float, float, int, int]] = []
    leafPolygonIDs: List[int] = []

    def _walk(node: Node) -> None:
        idx = len(nodes)
        bbox = node.bbox
        nodes.append((bbox.xMin, bbox.yMin, bbox.zMin, bbox.xMax, bbox.yMax, bbox.zMax, 0, 0))

        if node.isLeaf:
            offset = len(leafPolygonIDs)
            for polygon in node.polygons:
                leafPolygonIDs.append(polygonToTriangleID[id(polygon)])
            count = len(node.polygons)
            nodes[idx] = (
                bbox.xMin, bbox.yMin, bbox.zMin, bbox.xMax, bbox.yMax, bbox.zMax,
                count, offset,
            )
            return

        children = node.children
        if len(children) >= 1:
            _walk(children[0])
        rightOffset = int(NO_CHILD)
        if len(children) >= 2:
            rightOffset = len(nodes)
            _walk(children[1])
        # Internal node: polygonCount == 0; offset = rightChildOffset (or NO_CHILD).
        nodes[idx] = (
            bbox.xMin, bbox.yMin, bbox.zMin, bbox.xMax, bbox.yMax, bbox.zMax,
            0, rightOffset,
        )

    _walk(partition.root)
    return TreeNodeCL(nodes), LeafPolygonsCL(leafPolygonIDs)


def collectPolygonsFromLeaves(partition: SpacePartition) -> List[Polygon]:
    """Helper used by tests: list of every polygon as referenced from leaves (with duplicates)."""
    polygons: List[Polygon] = []

    def _walk(node: Node) -> None:
        if node.isLeaf:
            polygons.extend(node.polygons)
            return
        for child in node.children:
            _walk(child)

    _walk(partition.root)
    return polygons
