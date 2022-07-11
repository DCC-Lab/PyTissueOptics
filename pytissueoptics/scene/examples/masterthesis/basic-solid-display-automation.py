from pytissueoptics import *
import numpy as np

sphere = Sphere(radius=0.5, position=Vector(0, 0, 0))
solids = [sphere]
viewer = MayaviViewer()

ss = viewer.add(*solids, representation="surface", color=(0.8125, 0.8359, 0.89844))
scene = ss[0].parent.parent.parent.parent
scene.scene.background = (1, 1, 1)
s = ss[0]
lut = [[0.8125*255, 0.8359*255, 0.89844*255, i] for i in range(255)]
s.module_manager.scalar_lut_manager.lut.table = np.array(lut)
s.actor.property.opacity = 1
s.actor.property.edge_visibility = True
s.actor.property.line_width = 0.5
s.actor.property.edge_color = (25 / 255, 152 / 255, 158 / 255)
s.actor.property.interpolation = "flat"

viewer.show()
