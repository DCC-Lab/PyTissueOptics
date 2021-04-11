
photons = Photons()

d = getScatterindDistances()
(theta, phi) = getScatteringAngles()

directions = photons.scatteringDirections(theta, phi)
intersects = photons.intersections(direction, d) # [FresnelIntersect] or None

with photons.intersections(direction, d) as intersections:


with Intersecting(photons) as intersectingPhotons:





