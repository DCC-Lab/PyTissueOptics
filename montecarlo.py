import numpy as np
from vector import *
from material import *
from photon import *
from object import *

if __name__ == "__main__":
    mat    = Material(mu_s=30, mu_a = 0.5, g = 0.8)
    stats  = Stats(min = (-2, -2, -2), max = (2, 2, 2), size = (41,41,41))

    tissue = Object(material=mat, stats=stats)
    tissue.propagateManyPhotons(N=1000, showProgressEvery=100)
    tissue.report()
