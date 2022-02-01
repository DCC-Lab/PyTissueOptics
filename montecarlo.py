from pytissueoptics import *


world = World()

mat = Material(mu_s=2, mu_a=2, g=0.8, index=1.0)
stats = Stats(min=(-2, -2, -2), max=(2, 2, 2), size=(50, 50, 50))
source = PencilSource(maxCount=1000, direction=Vector(0, 0, 1))
tissue = Layer(thickness=2, material=mat, stats=stats)

world.place(source, position=Vector(0, 0, -1))
world.place(tissue, position=Vector(0, 0, -1))

world.simpleCompute(stats=stats)
world.report()
