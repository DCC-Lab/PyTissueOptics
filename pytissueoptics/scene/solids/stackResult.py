
class StackResult:
    def __init__(self, shape, position, vertices, surfaces, interfaces, primitive):
        """ Domain DTO to help creation of cuboid stacks. """
        self.shape = shape
        self.position = position
        self.vertices = vertices
        self.surfaces = surfaces
        self.interfaces = interfaces
        self.primitive = primitive
