
class StackResult:
    def __init__(self, shape, position, vertices, surfaces, primitive):
        """ Domain DTO to help creation of cuboid stacks. """
        self.shape = shape
        self.position = position
        self.vertices = vertices
        self.surfaces = surfaces
        self.primitive = primitive
