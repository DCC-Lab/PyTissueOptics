from pytissueoptics.scene.geometry.tree.kdtree import KDNode


class Splitter:
    def calculateSplitLine(self, node: KDNode):
        pass


class CentroidSplitter(Splitter):
    def calculateSplitLine(self, node: KDNode):
        average = 0
        for triangle in node.triangles:
            if node.axis == "x":
                average += triangle.globalCentroid.x
            elif node.axis == "y":
                average += triangle.globalCentroid.y
            elif node.axis == "z":
                average += triangle.globalCentroid.z
        splitLine = average / len(node.triangles)
        return splitLine


class BinarySplitter(Splitter):
    def calculateSplitLine(self, node):
        boundaryMin = node.boundingBox[node.axis.axis].min
        boundaryMax = node.boundingBox[node.axis.axis].max
        return boundaryMin + (boundaryMax-boundaryMin)/2
