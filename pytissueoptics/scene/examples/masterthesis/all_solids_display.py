from pytissueoptics import *
import numpy as np

cuboid = Cuboid(a=1, b=1.5, c=1, position=Vector(-2.75, 0, 0))
cube = Cube(1, position=Vector(-1.25, 0, 0))
sphere = Sphere(radius=0.5, position=Vector(0, 0, 0))
ellipsoid = Ellipsoid(a=0.8, b=0.5, c=0.5, position=Vector(1.5, 0, 0))
cylinder = Cylinder(0.5, 1, position=Vector(3, -0.5, 0))
cylinder.rotate(-90)
cone = Cone(0.5, 1, position=Vector(4.25, -0.5, 0))
cone.rotate(-90)

solids = [cuboid, cube, sphere, ellipsoid, cone, cylinder]
viewer = MayaviViewer()
ss = viewer.add(*solids, representation="surface", color=(0.8125, 0.8359, 0.89844))

for solid in solids:
    solid.translateBy(Vector(0, 2, 0))
sw = viewer.add(*solids, representation="wireframe", colormap='gray', lineWidth=1.5, reverseColormap=True)

scene = ss[0].parent.parent.parent.parent
scene.scene.background = (1, 1, 1)

for s in ss:
    s.actor.property.color = (0.8125, 0.8359, 0.89844)
    s.actor.property.opacity = 1
    s.actor.property.edge_visibility = False
    s.actor.property.line_width = 0.5
    s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s.actor.property.interpolation = "flat"

for s in sw:
    s.actor.property.color = (0.8125, 0.8359, 0.89844)
    s.actor.property.opacity = 1
    s.actor.property.edge_visibility = True
    s.actor.property.line_width = 5
    s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    lut = [[25, 152, 158, i] for i in range(255)]
    s.module_manager.scalar_lut_manager.lut.table = np.array(lut)
    s.actor.property.interpolation = "flat"

viewer.show()