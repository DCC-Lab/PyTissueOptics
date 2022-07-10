TITLE = "Load a .obj wavefront file"

DESCRIPTION = """ """


def exampleCode():
    from pytissueoptics.scene import loadSolid, MayaviViewer, Sphere, Cylinder, Vector
    import numpy as np

    #solid = loadSolid("/Users/marc-andrevigneault/Downloads/models/rat.obj")
    solid = Sphere(order=1)
    solid.scale(10)
    print(len(solid.getPolygons()))
    viewer = MayaviViewer()
    s1 = viewer.add(solid, representation="surface", constantColor=False, color=(0.8125, 0.8359, 0.89844))[0]
    viewer.add()
    scene = s1.parent.parent.parent.parent
    scene.scene.background = (1, 1, 1)
    s1.actor.property.color = (0.8125, 0.8359, 0.89844)
    s1.actor.property.opacity = 0.5
    s1.actor.property.edge_visibility = True
    s1.actor.property.line_width = 0.5
    s1.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s1.actor.property.interpolation = "flat"

    # LINE
    # solid2 = Cylinder(position=Vector(-10, -10, -10), height=30, radius=0.05)
    # solid2.rotate(0, 45, 45)
    # s2 = viewer.add(solid2)[0]
    # s2.actor.property.color = (0, 0, 0)
    # s2.actor.property.line_width = 0.05
    # s2.actor.property.edge_color = (0, 0, 0)
    # s2.actor.property.interpolation = "flat"

    # SECOND SPHERE
    solid2 = Sphere(order=1)
    solid2.scale(10)
    s2 = viewer.add(solid2)[0]
    s2.actor.property.representation = "wireframe"
    s2.module_manager.scalar_lut_manager.lut.table = np.array([[25, 152, 158, 255]]*255)
    s2.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s2.actor.property.edge_visibility = True
    s2.actor.property.line_width = 5
    s2.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s2.actor.property.interpolation = "flat"

    # SPHERE 3
    solid3 = Sphere(order=1)
    solid3.scale(10)
    s3 = viewer.add(solid2)[0]
    s3.actor.property.representation = "points"
    s3.module_manager.scalar_lut_manager.lut.table = np.array([[25, 152, 158, 255]] * 255)
    s3.actor.property.color = (25 / 255, 152 / 255, 158 / 255)
    s3.actor.property.interpolation = "flat"

    viewer.show()


if __name__ == "__main__":
    exampleCode()
