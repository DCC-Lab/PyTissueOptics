from .surface import *
from .source import *


class Geometry:
    def __init__(self, material=None, stats=None, label=""):
        self.material = material
        self.origin = Vector(0, 0, 0)
        self.stats = stats
        self.surfaces = []
        self.label = label
        self.inputWeight = 0

        self.epsilon = 1e-5
        self.startTime = None  # We are not calculating anything
        self.center = None

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        self.scoreWhenStarting(photon)
        d = 0
        while photon.isAlive and self.contains(photon.r):
            # Pick distance to scattering point
            if d <= 0:
                d = self.material.getScatteringDistance(photon)

            intersection = self.nextExitInterface(photon.r, photon.ez, d)

            if intersection is None:
                # If the scattering point is still inside, we simply move
                # Default is simply photon.moveBy(d) but other things 
                # would be here. Create a new material for other behaviour
                photon.moveBy(d)
                d = 0
                # Interact with volume: default is absorption only
                # Default is simply absorb energy. Create a Material
                # for other behaviour
                delta = photon.weight * self.material.albedo
                photon.decreaseWeightBy(delta)
                self.scoreInVolume(photon, delta)

                # Scatter within volume
                theta, phi = self.material.getScatteringAngles(photon)
                photon.scatterBy(theta, phi)
            else:
                # If the photon crosses an interface, we move to the surface
                photon.moveBy(d=intersection.distance)

                # Determine if reflected or not with Fresnel coefficients
                if intersection.isReflected():
                    # reflect photon and keep propagating
                    photon.reflect(intersection)
                    photon.moveBy(d=1e-3)  # Move away from surface
                    d -= intersection.distance
                else:
                    # transmit, score, and leave
                    photon.refract(intersection)
                    self.scoreWhenExiting(photon, intersection.surface)
                    photon.moveBy(d=1e-3)  # We make sure we are out
                    break

            # And go again    
            photon.roulette()

        # Because the code will not typically calculate millions of photons, it is
        # inexpensive to keep all the propagated photons.  This allows users
        # to go through the list after the fact for a calculation of their choice
        self.scoreWhenFinal(photon)
        photon.transformFromLocalCoordinates(self.origin)

    def contains(self, position) -> bool:
        """ The base object is infinite. Subclasses override this method
        with their specific geometry. 

        It is important that this function be efficient: it is called
        very frequently. See implementations for Box, Sphere and Layer
        """
        return True

    def validateGeometrySurfaceNormals(self):
        manyPhotons = IsotropicSource(maxCount = 10000)
        assert(self.contains(self.center))
        maxDist = 1000000
        for photon in manyPhotons:
            direction = Vector(photon.ez)
            origin = Vector(self.center)
            final = origin + direction*maxDist

            # We trace a line from the center to far away.
            intersect = self.nextEntranceInterface(position=origin, direction=direction, distance=maxDist)
            assert(intersect is None) # Because we are leaving, not entering

            intersect = self.nextExitInterface(position=origin, direction=direction, distance=maxDist)
            assert(intersect is not None) 
            assert(intersect.surface.contains(self.center + intersect.distance*direction))
            assert(intersect.indexIn == intersect.surface.indexInside)
            assert(intersect.indexOut == 1.0)
            assert(intersect.geometry == self)

            # We trace a line from far away to the center
            origin = final
            direction = -direction

            intersect = self.nextExitInterface(position=origin, direction=direction, distance=maxDist)
            assert(intersect is None) # Because we are entering, not leaving

            intersect = self.nextEntranceInterface(position=origin, direction=direction, distance=maxDist)
            assert(intersect is not None)
            assert(intersect.surface.contains(origin + intersect.distance*direction))
            assert(intersect.indexIn == intersect.surface.indexOutside)
            assert(intersect.indexOut == intersect.surface.indexInside)
            assert(intersect.geometry == self)

    def nextExitInterface(self, position, direction, distance) -> FresnelIntersect:
        """ Is this line segment from position to distance*direction leaving
        the object through any surface elements? Valid only from inside the object.
        
        This function is a very general function
        to find if a photon will leave the object.  `contains` is called
        repeatedly, is geometry-specific, and must be high performance. 
        It may be possible to write a specialized version for a subclass,
        but this version will work by default for all objects and is 
        surprisingly efficient.
        """

        finalPosition = Vector.fromScaledSum(position, direction, distance)
        if self.contains(finalPosition):
            return None

        # At this point, we know we will cross an interface: position is inside
        # finalPosition is outside.
        wasInside = True
        finalPosition = Vector(position)  # Copy
        delta = 0.1

        while abs(delta) > 0.00001:
            finalPosition += delta * direction
            isInside = self.contains(finalPosition)

            if isInside != wasInside:
                delta = -delta * 0.5

            wasInside = isInside

        for surface in self.surfaces:
            surfaceNormal = surface.normal(finalPosition)
            if surfaceNormal is None:
                return None

            if surfaceNormal.dot(direction) > 0:
                if surface.contains(finalPosition):
                    distanceToSurface = (finalPosition - position).abs()
                    return FresnelIntersect(direction, surface, distanceToSurface)

        return None

    def nextEntranceInterface(self, position, direction, distance) -> FresnelIntersect:
        """ Is this line segment from position to distance*direction crossing
        any surface elements of this object? Valid from outside the object.

        This will be very slow: going through all elements to check for
        an intersection is abysmally slow
        and increases linearly with the number of surface elements
        There are tons of strategies to improve this (axis-aligned boxes,
        oriented boxes but most importantly KDTree and OCTrees).
        It is not done here, we are already very slow: what's more slowdown
        amongst friends? """

        minDistance = distance
        intersectSurface = None
        for surface in self.surfaces:
            surfaceNormal = surface.normal()
            if surfaceNormal is None:
                continue
            elif surfaceNormal.dot(direction) > 0:
                continue
            # Going inward, against surface normal
            isIntersecting, distanceToSurface, positionOnSurface = surface.intersection(position, direction, distance)
            if isIntersecting and distanceToSurface < minDistance:
                intersectSurface = surface
                minDistance = distanceToSurface

        if intersectSurface is None:
            return None
        return FresnelIntersect(direction, intersectSurface, minDistance, self)

    @staticmethod
    def isReflected(photon, surface) -> bool:
        R = photon.fresnelCoefficient(surface)
        if np.random.random() < R:
            return True
        return False

    def scoreWhenStarting(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenStarting(photon)

    def scoreInVolume(self, photon, delta):
        if self.stats is not None:
            self.stats.scoreInVolume(photon, delta)

    def scoreWhenExiting(self, photon, surface):
        if self.stats is not None:
            self.stats.scoreWhenCrossing(photon, surface)

    def scoreWhenEntering(self, photon, surface):
        return

    def scoreWhenFinal(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenFinal(photon)

    def report(self, totalSourcePhotons):
        print("{0}".format(self.label))
        print("=====================\n")
        print("Geometry and material")
        print("---------------------")
        print(self)

        print("\nPhysical quantities")
        print("---------------------")
        if self.stats is not None:
            totalWeightAcrossAllSurfaces = 0
            for i, surface in enumerate(self.surfaces):
                totalWeight = self.stats.totalWeightCrossingPlane(surface)
                print("Transmittance [{0}] : {1:.1f}% of propagating light".format(surface,
                                                                                   100 * totalWeight / self.stats.inputWeight))
                print("Transmittance [{0}] : {1:.1f}% of total power".format(surface,
                                                                             100 * totalWeight / totalSourcePhotons))
                totalWeightAcrossAllSurfaces += totalWeight

            print("Absorbance : {0:.1f}% of propagating light".format(
                100 * self.stats.totalWeightAbsorbed() / self.stats.inputWeight))
            print("Absorbance : {0:.1f}% of total power".format(
                100 * self.stats.totalWeightAbsorbed() / totalSourcePhotons))

            totalCheck = totalWeightAcrossAllSurfaces + self.stats.totalWeightAbsorbed()
            print("Absorbance + Transmittance = {0:.1f}%".format(100 * totalCheck / self.stats.inputWeight))

            self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            if len(self.surfaces) != 0:
                self.stats.showSurfaceIntensities(self.surfaces, maxPhotons=totalSourcePhotons)

    def __repr__(self):
        return "{0}".format(self)

    def __str__(self):
        string = "'{0}' {2} with surfaces {1}\n".format(self.label, self.surfaces, self.origin)
        string += "{0}".format(self.material)
        return string


class Box(Geometry):
    def __init__(self, size, material, stats=None, label="Box"):
        super(Box, self).__init__(material, stats, label)
        self.size = size
        self.surfaces = [-XYPlane(atZ=-self.size[2] / 2, description="Front"),
                         XYPlane(atZ=self.size[2] / 2, description="Back"),
                         -YZPlane(atX=-self.size[0] / 2, description="Left"),
                         YZPlane(atX=self.size[0] / 2, description="Right"),
                         -ZXPlane(atY=-self.size[1] / 2, description="Bottom"),
                         ZXPlane(atY=self.size[1] / 2, description="Top")]
        self.center = ConstVector(0,0,0)

    def contains(self, localPosition) -> bool:
        if abs(localPosition.z) > self.size[2] / 2 + self.epsilon:
            return False
        if abs(localPosition.y) > self.size[1] / 2 + self.epsilon:
            return False
        if abs(localPosition.x) > self.size[0] / 2 + self.epsilon:
            return False

        return True


class Cube(Box):
    def __init__(self, side, material, stats=None, label="Cube"):
        super(Cube, self).__init__((side, side, side), material, stats, label)


class Layer(Geometry):
    def __init__(self, thickness, material, stats=None, label="Layer"):
        super(Layer, self).__init__(material, stats, label)
        self.thickness = thickness
        self.surfaces = [-XYPlane(atZ=0, description="Front"),
                         XYPlane(atZ=self.thickness, description="Back")]
        self.center = ConstVector(0,0,thickness/2)

    def contains(self, localPosition) -> bool:
        if localPosition.z < -self.epsilon:
            return False
        if localPosition.z > self.thickness + self.epsilon:
            return False

        return True

    def nextExitInterface(self, position, direction, distance) -> FresnelIntersect:
        finalPosition = Vector.fromScaledSum(position, direction, distance)
        if self.contains(finalPosition):
            return None
        # assert(self.contains(position) == True)

        if direction.z > 0:
            d = (self.thickness - position.z) / direction.z
            if d <= distance:
                s = self.surfaces[1]
                intersect = FresnelIntersect(direction, self.surfaces[1], d, geometry=self) 
                # assert(s.indexInside)
                # print(s.indexInside, s.indexOutside)
                # print(intersect.indexIn, intersect.indexOut)
                return intersect
        elif direction.z < 0:
            d = - position.z / direction.z
            if d <= distance:
                s = self.surfaces[0]
                intersect = FresnelIntersect(direction, self.surfaces[0], d, geometry=self) 
                # print(s.indexInside, s.indexOutside)
                # print(intersect.indexIn, intersect.indexOut)
                return intersect


        return None

    def stack(self, layer):
        return


class SemiInfiniteLayer(Geometry):
    """ This class is actually a bad idea: the photons don't exit
    on the other side and will just take a long time to propagate.
    It is better to use a finite layer with a thickness a bit larger
    than what you are interested in."""

    def __init__(self, material, stats=None, label="Semi-infinite layer"):
        super(SemiInfiniteLayer, self).__init__(material, stats, label)
        self.surfaces = [-XYPlane(atZ=0, description="Front")]
        self.center = ConstVector(0,0,1)

    def contains(self, localPosition) -> bool:
        if localPosition.z < -self.epsilon:
            return False

        return True

    def nextExitInterface(self, position, direction, distance) -> FresnelIntersect:
        finalPosition = position + distance * direction
        if self.contains(finalPosition):
            return None
        assert(self.contains(position) == True)

        if direction.z < 0:
            d = - position.z / direction.z
            if d <= distance:
                return FresnelIntersect(direction, self.surfaces[0], d, geometry=self) 

        return None


class Sphere(Geometry):
    def __init__(self, radius, material, stats=None, label="Sphere"):
        super(Sphere, self).__init__(material, stats, label)
        self.radius = radius
        self.surfaces = [ Conic(R=radius, kappa=0, normal=-zHat, diameter=2*radius, description="Front"),
                          Conic(R=-radius, kappa=0, normal=zHat, diameter=2*radius, description="Back")]
        self.center = ConstVector(0,0,0)

    def contains(self, localPosition) -> bool:
        if localPosition.x*localPosition.x+localPosition.y*localPosition.y > self.radius*self.radius:
            return False

        return True

    def nextExitInterface(self, position, direction, distance) -> FresnelIntersect:
        """ Is this line segment from position to distance*direction leaving
        the object through any surface elements? Valid only from inside the object.
        
        """

        for surface in self.surfaces:
            isIntersecting, distanceToSurface, pointOnSurface = surface.intersection(position, direction, distance)

            if isIntersecting:
                return FresnelIntersect(direction, surface, distanceToSurface)

        return None


class KleinBottle(Geometry):
    def __init__(self, position, material, stats=None):
        super(KleinBottle, self).__init__(position, material, stats)

    def contains(self, localPosition) -> bool:
        raise NotImplementedError()
