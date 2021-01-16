from vector import *
from material import *
from photon import *
from object import *

if __name__ == "__main__":
    mat    = Material(mu_s=30, mu_a = 0.5, g = 0.8)
    stats  = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))
    source = LaserSource(position=Vector(0,0,0), direction=Vector(0,0,1), maxCount=10000)
    #tissue = Cube(side=2, material=mat, stats=stats)
    tissue = Sphere(radius=1, material=mat, stats=stats)
    tissue.propagateMany(source, showProgressEvery=100)
    tissue.report()
