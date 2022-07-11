from pytissueoptics import *
import numpy as np

solids = []
for i in range(4):
    s = Sphere(radius=0.5, order=i, position=Vector(i+(i*0.25), 0, 0))
    solids.append(s)

viewer = MayaviViewer()
s1 = viewer.add(*solids, representation="surface", colormap='gray')

for solid in solids:
    solid.translateBy(Vector(0, 1.5, 0))
s2 = viewer.add(*solids, representation="surface", colormap='gray')

scene = s1[0].parent.parent.parent.parent
scene.scene.background = (1, 1, 1)

lut = [[0.8125*255, 0.8359*255, 0.89844*255, 255] for i in range(255)]
for s in s1:
    s.module_manager.scalar_lut_manager.lut.table = np.array(lut)
    s.actor.property.opacity = 1
    s.actor.property.edge_visibility = False
    s.actor.property.line_width = 0.5
    s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s.actor.property.interpolation = "phong"

for s in s2:
    s.module_manager.scalar_lut_manager.lut.table = np.array(lut)
    s.actor.property.opacity = 1
    s.actor.property.edge_visibility = False
    s.actor.property.line_width = 0.5
    s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
    s.actor.property.interpolation = "flat"

viewer.show()