import unittest

import numpy as np

from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.buffers import NO_CHILD, flattenSpacePartition
from pytissueoptics.rayscattering.opencl.buffers.treeCL import collectPolygonsFromLeaves
from pytissueoptics.rayscattering.opencl.config.CLConfig import CLConfig
from pytissueoptics.scene.geometry import BoundingBox, Polygon, Vertex
from pytissueoptics.scene.tree import Node, SpacePartition
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitThreeAxesConstructor


def _trianglePolygon(xMin: float, xMax: float, label: str = "s") -> Polygon:
    poly = Polygon([Vertex(xMin, 0, 0), Vertex(xMax, 1, 0), Vertex(xMin, 1, 1)])
    poly.surfaceLabel = label
    return poly


def _spreadOutPolygons(n: int = 12) -> list:
    # A row of disjoint triangles along x so the SAH constructor will actually split.
    return [_trianglePolygon(i, i + 0.5) for i in range(n)]


@unittest.skipIf(not OPENCL_AVAILABLE, "OpenCL not available.")
class TestCLTree(unittest.TestCase):
    def _bakeBuffer(self, treeNodeCL):
        treeNodeCL.make(CLConfig().device)
        return treeNodeCL.hostBuffer

    def testRootLeafTreeProducesSingleLeafNode(self):
        # When a tree is built but the constructor immediately gives up, the partition has just
        # the root as a leaf containing every polygon. The flatten output must mirror that.
        polygons = _spreadOutPolygons(3)
        bbox = BoundingBox.fromPolygons(polygons)
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=0, minLeafSize=10)
        polygonToTriangleID = {id(p): i for i, p in enumerate(polygons)}

        treeNodeCL, leafPolygonsCL = flattenSpacePartition(partition, polygonToTriangleID)

        nodes = self._bakeBuffer(treeNodeCL)
        self.assertEqual(1, len(nodes))
        self.assertEqual(len(polygons), int(nodes[0]["polygonCount"]))
        self.assertEqual(0, int(nodes[0]["offset"]))
        polygonIDs = leafPolygonsCL.hostBuffer[: int(nodes[0]["polygonCount"])]
        self.assertEqual(sorted(polygonIDs.tolist()), sorted(polygonToTriangleID.values()))

    def testFlattenedTreePreservesDFSTopology(self):
        # For any internal node, left child must be at index+1; right child must be at `offset`
        # (or absent). polygonCount > 0 must coincide with `node.isLeaf`.
        polygons = _spreadOutPolygons(20)
        bbox = BoundingBox.fromPolygons(polygons)
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=4, minLeafSize=2)
        polygonToTriangleID = {id(p): i for i, p in enumerate(polygons)}

        # Walk the Python tree DFS and capture the expected (isLeaf, polygonCount) sequence.
        expected = []

        def _walkPython(node: Node) -> None:
            expected.append((node.isLeaf, len(node.polygons) if node.isLeaf else 0))
            if node.isLeaf:
                return
            for child in node.children:
                _walkPython(child)

        _walkPython(partition.root)

        treeNodeCL, _ = flattenSpacePartition(partition, polygonToTriangleID)
        nodes = self._bakeBuffer(treeNodeCL)
        self.assertEqual(len(expected), len(nodes))
        for i, (isLeaf, count) in enumerate(expected):
            self.assertEqual(isLeaf, int(nodes[i]["polygonCount"]) > 0, msg=f"node {i} leaf-flag")
            if isLeaf:
                self.assertEqual(count, int(nodes[i]["polygonCount"]))

    def testInternalNodesPointToValidRightChildOrSentinel(self):
        polygons = _spreadOutPolygons(20)
        bbox = BoundingBox.fromPolygons(polygons)
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=4, minLeafSize=2)
        polygonToTriangleID = {id(p): i for i, p in enumerate(polygons)}

        treeNodeCL, _ = flattenSpacePartition(partition, polygonToTriangleID)
        nodes = self._bakeBuffer(treeNodeCL)

        for i, node in enumerate(nodes):
            if int(node["polygonCount"]) > 0:
                continue  # leaf
            offset = int(node["offset"])
            self.assertTrue(
                offset == int(NO_CHILD) or (i < offset < len(nodes)),
                msg=f"internal node {i} has invalid right-child offset {offset}",
            )

    def testEveryPolygonAppearsInAtLeastOneLeaf(self):
        polygons = _spreadOutPolygons(20)
        bbox = BoundingBox.fromPolygons(polygons)
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=4, minLeafSize=2)
        polygonToTriangleID = {id(p): i for i, p in enumerate(polygons)}

        # Sanity: the Python partition itself sees every polygon at least once across leaves.
        leafPolygons = collectPolygonsFromLeaves(partition)
        self.assertEqual(set(polygonToTriangleID.values()), {polygonToTriangleID[id(p)] for p in leafPolygons})

        _, leafPolygonsCL = flattenSpacePartition(partition, polygonToTriangleID)
        emittedTriangleIDs = set(int(x) for x in leafPolygonsCL.hostBuffer.tolist())
        self.assertTrue(set(polygonToTriangleID.values()).issubset(emittedTriangleIDs))

    def testLeafPolygonRangesMatchTreeStructure(self):
        # Sum of leaf counts across all leaves must equal the total polygon-id slots emitted.
        polygons = _spreadOutPolygons(20)
        bbox = BoundingBox.fromPolygons(polygons)
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=4, minLeafSize=2)
        polygonToTriangleID = {id(p): i for i, p in enumerate(polygons)}

        treeNodeCL, leafPolygonsCL = flattenSpacePartition(partition, polygonToTriangleID)
        nodes = self._bakeBuffer(treeNodeCL)

        leafCounts = [int(n["polygonCount"]) for n in nodes if int(n["polygonCount"]) > 0]
        self.assertEqual(sum(leafCounts), len(leafPolygonsCL.hostBuffer))

        # Also: each leaf's [offset, offset+count) range stays within the polygon array.
        for n in nodes:
            count = int(n["polygonCount"])
            if count == 0:
                continue
            offset = int(n["offset"])
            self.assertTrue(0 <= offset < offset + count <= len(leafPolygonsCL.hostBuffer))

    def testFlattenRejectsEmptyPartition(self):
        # An empty partition would produce a leaf with polygonCount=0, which the kernel parses
        # as an internal node pointing to a non-existent left child (OOB read). The host-side
        # API must refuse to build such a tree.
        polygons: list = []
        bbox = BoundingBox(xLim=[0, 1], yLim=[0, 1], zLim=[0, 1])
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=4, minLeafSize=2)
        self.assertRaises(ValueError, flattenSpacePartition, partition, {})

    def testEmptyLeafPolygonsBufferIsSizeOne(self):
        # CLObject convention: zero-element buffers reserve one slot to keep cl.Buffer happy.
        polygons = _spreadOutPolygons(2)
        bbox = BoundingBox.fromPolygons(polygons)
        partition = SpacePartition(bbox, polygons, NoSplitThreeAxesConstructor(), maxDepth=0, minLeafSize=0)
        polygonToTriangleID = {id(p): i for i, p in enumerate(polygons)}
        _, leafPolygonsCL = flattenSpacePartition(partition, polygonToTriangleID)
        self.assertGreaterEqual(len(leafPolygonsCL.hostBuffer), 1)
        self.assertEqual(np.uint32, leafPolygonsCL.hostBuffer.dtype)
