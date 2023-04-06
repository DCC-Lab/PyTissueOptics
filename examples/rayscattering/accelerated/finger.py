import os
import sys

modulePath = os.path.abspath(__file__ + 4 * '/..')
sys.path.append(modulePath)

from pytissueoptics import *

"""
Simple and unrealistic finger model for testing purposes

References:
 Bone mu_s & mu_a: https://www.researchgate.net/publication/336109612_Optical_Properties_on_Bone_Analysis_An_Approach_to_Biomaterials/fulltext/5d8eabcf92851c33e942f943/Optical-Properties-on-Bone-Analysis-An-Approach-to-Biomaterials.pdf
 RI: https://www.researchgate.net/figure/Refractive-index-properties-of-materials-encountered-during-typical-imaging-of_fig9_296486438
 Skin *reduced* mu_s`(λ) : https://iopscience.iop.org/article/10.1088/0031-9155/58/11/R37
 
 anisotropy (fig8) : https://iopscience.iop.org/article/10.1088/0031-9155/58/11/R37/pdf
 subcutaneous fat g (1200nm+): https://ars.els-cdn.com/content/image/1-s2.0-S1466856413000787-gr3.jpg
 fat mu_a (fig13 & 16) (Simpson, 600-1000nm): https://iopscience.iop.org/article/10.1088/0031-9155/58/11/R37
 geometry: ...
"""
# fixme: default IPP test of 1000 photons will take a long time (20s on my laptop).
#  can be changed manually in /rayscattering/opencl/config.json if you want.

IR = True  # around 1100 nm
boneMat = ScatteringMaterial(mu_s=600, mu_a=1.5, g=0.92, n=1.55)  # 1000 nm
fatMat = ScatteringMaterial(mu_s=150, mu_a=0.13, g=0.93, n=1.46)  # mu_a @ 1100 nm
skinMat = ScatteringMaterial(mu_s=100, mu_a=0.15, g=0.95, n=1.36)  # g @ 1100 nm

if not IR:  # around 550 nm
    # todo: find fat g in visible
    # todo: mu_s
    # big difference between papers. Some might be talking about reduced mu_s' without mentioning it
    # S.L.Jacques results all show mu_s' typically around 20 @ 1000nm and 40 around 500nm
    #  Considering g typically between 0.8 and 0.95, this sets mu_s between 5x and 20x mu_s'
    #   => mu_s in the range of [100, 400] @ 1000nm and [200, 800] @ 500nm
    # jacques: skin mu_s in visible is around 200 for g around 0.75 @ 488nm
    boneMat = ScatteringMaterial(mu_s=600, mu_a=2.2, g=0.92, n=1.55)  # 700 nm
    fatMat = ScatteringMaterial(mu_s=300, mu_a=0.5, g=0.85, n=1.46)  # mu_a estimated with fig 16 @ 550 nm, g guessed
    skinMat = ScatteringMaterial(mu_s=200, mu_a=0.5, g=0.8, n=1.36)  # g @ 550 nm

epidermis = Ellipsoid(0.9, 3, 0.9, order=3, material=ScatteringMaterial(mu_s=20, mu_a=2, g=0.95, n=1.36), label="epidermis")
dermis = Ellipsoid(0.85, 2.95, 0.85, order=3, material=ScatteringMaterial(mu_s=20, mu_a=2, g=0.95, n=1.36), label="dermis")
fat = Ellipsoid(0.8, 2.9, 0.8, order=3, material=fatMat, label="fat")
bone = Cylinder(0.25, 5, material=boneMat, label="bone")
bone.rotate(90)

scene = ScatteringScene([epidermis, dermis, fat, bone])
logger = EnergyLogger(scene)

source = DirectionalSource(position=Vector(0, 0, -2), direction=Vector(0, 0, 1), N=10000, useHardwareAcceleration=True,
                           diameter=0.1)

source.propagate(scene, logger)

viewer = Viewer(scene, source, logger)
viewer.reportStats()
colormap = ["Blues", "Reds"][IR]
viewer.show2D(View2DProjectionX(limits=((-1, 1), (-1.5, 1.5)), binSize=0.005), colormap=colormap)
# viewer.show3D()
