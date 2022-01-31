from python_graphics_engine.geometry import Polygon, Vector


class Triangle(Polygon):
    def __init__(self, v1: Vector, v2: Vector, v3: Vector):
        super().__init__(vertices=[v1, v2, v3])

# todo:
#  add inside and outside material reference in each surface
#  make solid methods abstract
