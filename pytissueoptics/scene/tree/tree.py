from typing import List
from pytissueoptics.scene.geometry import BoundingBox, Vector
from pytissueoptics.scene.tree import TreeConstructor
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.solids import Cuboid
from pytissueoptics.scene.scene import Scene


class Tree:
    def __init__(self, scene: Scene, constructor: TreeConstructor, maxDepth=6, maxLeafSize=2):
        self._scene = scene
        self._maxDepth = maxDepth
        self._maxLeafSize = maxLeafSize
        self._constructor = constructor
        self._root = Node(scene=scene, maxDepth=maxDepth, maxLeafSize=maxLeafSize)
        self._constructor.growTree(self._root)

    def searchPoint(self, point: Vector) -> BoundingBox:
        return self._root.searchPoint(point)

    def searchRayIntersection(self, ray):
        return self._root.searchRayIntersection(ray)

    def _getBoundingBoxes(self) -> List[BoundingBox]:
        boundingBoxes = self._root.getLeafBoundingBoxes(bboxList=[])
        return boundingBoxes

    def getNodeCount(self):
        return self._root.getNodeCount()

    def getLeafCount(self):
        return self._root.getLeafCount()

    def getLeafBoundingBoxesAsCuboids(self) -> List[Cuboid]:
        cuboids = []
        for bbox in self._getBoundingBoxes():
            a = bbox.xMax-bbox.xMin
            b = bbox.yMax-bbox.yMin
            c = bbox.zMax-bbox.zMin
            cuboids.append(Cuboid(a=a, b=b, c=c, position=bbox.center))
        return cuboids
