from pytissueoptics.scene.geometry.tree.kdtree import KDNode


class Splitter:
    def calculateSplitLine(self, node: KDNode) -> float:
        raise NotImplementedError


class CentroidSplitter(Splitter):
    def calculateSplitLine(self, node: KDNode) -> float:
        average = 0
        for polygon in node.polygons:
            if node.axis == "x":
                average += polygon.centroid.x
            elif node.axis == "y":
                average += polygon.centroid.y
            elif node.axis == "z":
                average += polygon.centroid.z

        average = average / len(node.polygons)
        return average


class BinarySplitter(Splitter):
    def calculateSplitLine(self, node: KDNode) -> float:
        boundaryMin = node.boundingBox.getAxisLimit(node.axis, "min")
        boundaryMax = node.boundingBox.getAxisLimit(node.axis, "max")
        return (boundaryMin + boundaryMax) / 2
