from vector import *

class Object:
    def __init__(self, origin=Vector(0,0,0), material=Material(0,0,0)):
        self.origin = origin # Global coordinates
        self.material = material
        self.boudingBoxMin = None # Local coordinates
        self.boudingBoxMax = None # Local coordinates
        self.objects = []
        self.container = None

    def placeInto(self, container, position):
        # position in local coordinates of container
        self.translate(container.origin + position) 
        self.container = container
        container.append(self)

    def translateBy(self, position:Vector)
        self.boundingBoxMin = self.boundingBoxMin - self.origin + position 
        self.boundingBoxMax = self.boundingBoxMax - self.origin + position 
        self.origin += position

    def contains(self, position) -> bool:
        if position.x < self.boundingBoxMin.x or position.x > self.boundingBoxMax.x:
            return False
        elif position.y < self.boundingBoxMin.y or position.y > self.boundingBoxMax.y:
            return False
        elif position.z < self.boundingBoxMin.z or position.z > self.boundingBoxMax.z:
            return False

        return True

class Cube(Object):
    def __init__(self, side, origin=Vector(0,0,0), material=Material(0,0,0)):
        self.side = side
        Object.__init__(origin, material)
        self.boundingBoxMin = origin - Vector(side/2,side/2,side/2)
        self.boundingBoxMax = origin + Vector(side/2,side/2,side/2)

class InfiniteLayer(Object):
    def __init__(self, thickness, origin=Vector(0,0,0), material=Material(0,0,0)):
        self.thickness = thickness
        Object.__init__(origin, material)
        self.boundingBoxMin = origin + Vector(-1000,-1000,thickness)
        self.boundingBoxMax = origin + Vector(+1000,+1000,thickness)

    def contains(self, position) -> bool:
        if position.z < self.boundingBoxMin.z or position.z > self.boundingBoxMax.z:
            return False

        return True
