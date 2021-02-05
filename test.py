from pytissueoptics import *

# We pick a geometry
mat    = Material(mu_s=1, mu_a = 1, g = 0.8, index = 1.0)
tissue = Layer(thickness=2, material=mat)

position = Vector(0,0,-3)
direction = UnitVector(1, 1, 1)
maxDistance = 10

surface = XYPlane(atZ= 0, description="Front")
intersect, distance = surface.intersection(position=position, 
                                           direction=direction,
                                           maxDistance=maxDistance)

print(intersect, distance, direction, Vector.fromScaledSum(position, direction, distance))
# print(tissue.nextEntranceInterface(position=position, 
#                                    direction=direction,
#                                    distance=distance))