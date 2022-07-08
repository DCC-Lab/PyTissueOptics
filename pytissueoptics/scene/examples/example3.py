TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import loadSolid, MayaviViewer

    solid = loadSolid("/Users/marc-andrevigneault/Downloads/models/rat.obj")
    solid.scale(10)
    print(len(solid.getPolygons()))
    viewer = MayaviViewer()
    s1 = viewer.add(solid, representation="surface", constantColor=False, color=(0.8125, 0.8359, 0.89844))[0]
    viewer.add()
    scene = s1.parent.parent.parent.parent
    scene.scene.background = (1, 1, 1)
    s1.actor.property.color = (0.8125, 0.8359, 0.89844)
    s1.actor.property.edge_visibility = True
    s1.actor.property.line_width = 0.5
    s1.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s1.actor.property.interpolation = "flat"
    viewer.show()


if __name__ == "__main__":
    exampleCode()
