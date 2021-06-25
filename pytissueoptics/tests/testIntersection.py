import envtest  # modifies path
from pytissueoptics import *
from numpy import linspace, pi, sqrt, polyfit
from pytissueoptics.surface import AsphericSurface
import matplotlib.pyplot as plt

inf = float("+inf")

"""
I have a programming problem, and I return it to you as a programming challenge. I have found various solutions, but they are either 1) ugly, 2) don’t work perfectly all the time or 3) both.  I worked in PyTIssueOptics, branch curvedSurfaces at https://github.com/DCC-Lab/PyTissueOptics/tree/curvedSurfaces

Intersection of a finite quadratic surface with a line segment
=============================================

You can use the Vector class in PyTissueOptics for all vector calculations.
I have a line segment from point A to point B. It could be anywhere in space. I have a surface above the xy plane defined with z=f(x,y).  The function f(x,y) is quadratic and is the equation of a conic (R, kappa, typical):

def f(x,y):
    R = 10         # anything positive or negative
    kappa = -0.5   # values mostly between -1 and 1, but can be anything including zero
    r = np.sqrt(x*x+y*y)
    z = r*r/(R*(1+sqrt(1-(1+kappa)*r*r/R/R)))

    return z
It could also be extremely general and a polynomial of any degree (you can look at https://en.wikipedia.org/wiki/Aspheric_lens)


Requirements
==========

I want  a function :
def intersection(self, position, direction, maxDistance) -> (bool, float, Vector):
where position  = point A
direction is (OB-OA).normalized()
maxDistance == (OB-OA).abs()

The function must return:
True/False if the segment intersects
a float with the distance from point A
the exact point where they intersect (to within 1e-6).

Performance is a concern in general but at this point, I need a reference function that works all the time, regardless of speed.

Submit your solution in python!

"""
class TestIntersection(envtest.PyTissueTestCase):

    def testAbritraryF(self):
        origin = Vector(0,1,100)
        final = Vector(0,0,-1)
        d = final-origin
        pts = []
        for t in linspace(0,1,100):
            pts.append(self.line(origin, d, t))

        differences = []
        for p in pts:
            surfaceZ = self.f(p.x, p.y)

            if not isnan(surfaceZ):
                delta = p.z - surfaceZ
                differences.append(delta)

        self.assertTrue(min(differences) < 0)

    def line(self, o, d, t):
        return o + t*d

    # def f(self, x, y, R=35, diameter=25):
    #     try:
    #         r2 = x*x+y*y
    #         if r2 > diameter*diameter:
    #             return None
    #         value = np.sqrt(R*R-r2)
    #         if isnan(value):
    #             return None
    #         return value
    #     except:
    #         return None

    def f(self, x,y, R=35, kappa=0):
        try:
            r2 = x*x+y*y
            value =  r2/(R*(1+sqrt(1-(1+kappa)*r2/R/R)))

            if isnan(value):
                return None
            return value
        except:
            return None


    @envtest.skip("")
    def testIntersectionPoint(self):
        position = self.randomNegativeZVector()
        final = Vector(0,0,0.1)

        d = final-position
        distance = d.abs()
        direction = d.normalized()
        mustIntersect = False
        if position.abs() > 1 and final.abs() < 1:
            mustIntersect = True

        isIntersecting, d, current = self.algo1(position, direction, distance)
        # print(isIntersecting, d, current)
        # self.assertEqual(mustIntersect, isIntersecting,msg="{0} {1} {2} {3}".format(position.abs(), final.abs(), position, final))
        if isIntersecting:
            self.assertAlmostEqual( self.f(current.x, current.y) - current.z, 0, 5 )
            self.assertAlmostEqual( current.abs(), 1, 5)

    @envtest.skip("")
    def testIntersectionManyPoints(self):
        for i in range(1000):
            position = self.randomNegativeZVector() 
            final = Vector(0,0,1)

            d = final-position
            distance = d.abs()
            direction = d.normalized()
            print(d, distance, direction)
            self.assertTrue(direction.isUnitary)
            isIntersecting, d, current = self.algo1(position, direction, distance)
            
            self.assertTrue(isIntersecting, msg="{0} {1} {2}".format(position, final, current))
            if isIntersecting:
                self.assertAlmostEqual( self.f(current.x, current.y) - current.z, 0, 5 )
                self.assertAlmostEqual( current.abs(), 1, 5)

    def algo1(self, position, direction, maxDistance) -> (bool, float, Vector):
        wasBelow = None
        startingPointOnSurface = self.f(position.x, position.y)
        if startingPointOnSurface is not None:
            wasBelow = position.z < startingPointOnSurface

        if wasBelow is False:
            return (False, None, None)
        current = Vector(position)  # Copy
        delta = 0.1
        t = 0
        while abs(delta) > 0.0000001:
            t += delta
            current = position + t * direction * maxDistance
            surfaceZ = self.f(current.x, current.y)
            isBelow = None
            if surfaceZ is not None:
                isBelow = (current.z < surfaceZ)

            if wasBelow is None or isBelow is None:
                pass
            elif wasBelow != isBelow:
                delta = -delta * 0.5

            #print(position, current, wasBelow, isBelow, t, maxDistance)
            wasBelow = isBelow
            if t > 1.3 or t < -0.3:
                return (False, None, None)

        if t >= 0 and t <= 1.0:
            return (True, t * maxDistance, current)
        else:
            return (False, None, None)

    def randomVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2-1
        return Vector(x,y,z)

    def randomPositiveZVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2
        return Vector(x,y,z)

    def randomNegativeZVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = -random.random()*2
        return Vector(x,y,z)

    def randomVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( Vector(x,y,z) )

    def randomUnitVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( UnitVector(x,y,z) )

    # def testRandomIntersectionPoint(self):
    #     for i in range(100):

    #         position = self.randomNegativeZVector()+Vector(0,0,-1.1)
    #         final = Vector(0,0,0.5)
    #         d = final-position
    #         distance = d.abs()
    #         direction = d.normalized()

    #         finalPosition = Vector(position)
    #         surfaceZ = self.f(finalPosition.x, finalPosition.y)
    #         if surfaceZ is None:
    #             wasBelow = None
    #         else:
    #             wasBelow = (finalPosition.z < surfaceZ)

    #         delta = 0.1
    #         t = 0
    #         while abs(delta) > 0.000001:
    #             t += delta
    #             finalPosition = position + t * direction * distance
    #             surfaceZ = self.f(finalPosition.x, finalPosition.y)

    #             if surfaceZ is not None:
    #                 isBelow = (finalPosition.z < surfaceZ)

    #                 if wasBelow is None or isBelow is None:
    #                     pass # Keep going
    #                 elif isBelow != wasBelow:
    #                     delta = -delta * 0.5
    #                 else:
    #                     delta = 1.5*delta

    #                 wasBelow = isBelow

    #             if t > 1 and delta > 0:
    #                 finalPosition = None
    #                 break
    #             elif t < 0 and delta < 0:
    #                 finalPosition = None
    #                 break

    #         if finalPosition is not None:
    #             self.assertAlmostEqual( self.f(finalPosition.x, finalPosition.y) - finalPosition.z,0, 4)
    #             self.assertAlmostEqual( finalPosition.abs(), 1, 4)
    #         else:
    #             print(i, position, final, t)

    # def testRandomIntersectionPointAlgorithm2(self):
    #     ts = linspace(0,1,5)
    #     for i in range(10000):
    #         position = 2*self.randomPositiveZVector()
    #         final = Vector(0,0,0)
    #         direction = final-position

    #         xs = []
    #         ys = []
    #         for t in ts:
    #             pt = self.line(position, direction, t)
    #             surfaceZ = self.f(pt.x, pt.y, R=1)
    #             if surfaceZ is not None:
    #                 ys.append( pt.z-surfaceZ )
    #                 xs.append( t )
            
    #         (a,b,c) = polyfit(xs, ys, deg=2)
    #         delta = sqrt(b*b-4*a*c)
    #         if isnan(delta):
    #             self.fail("No")

    #         tm = (-b-delta)/2/a
    #         tp = (-b+delta)/2/a
    #         tmin = self.smallestValidT(tm, tp)


    #         self.assertIsNotNone(tmin)

    def smallestValidT(self, tm, tp):
        valid = []
        if tm >= 0 and tm <= 1:
            valid.append(tm)
        if tp >= 0 and tp <= 1:
            valid.append(tp)

        if len(valid) == 0:
            return None
        elif len(valid) == 1:
            return valid[0]
        elif len(valid) == 2:
            return min(valid)


        #     fullT = linspace(0,1,10)
        #     quad = [ a*t*t + b*t + c for t in fullT]
        #     plt.plot(xs,ys,'o')
        #     plt.plot(fullT,quad)
        # plt.show()

            # if finalPosition is not None:
            #     self.assertAlmostEqual( self.f(finalPosition.x, finalPosition.y) - finalPosition.z,0, 4)
            #     self.assertAlmostEqual( finalPosition.abs(), 1, 4)
            # else:
            #     print(i, position, final, t)

    # def testAlgo3(self):

    #     position = self.randomPositiveZVector()+Vector(0,0,1.1)
    #     final = Vector(0,0,0)
    #     direction = final-position
    #     surfacet0 = self.f(position.x, position.y)
    #     surfacet1 = self.f(final.x, final.y)

    #     if surfacet0 is None:

    #     t = 0
    #     delta = 0.1
    #     while abs(delta) > 0.000001:
    #         t += delta
    #         finalPosition = position + t * direction
    #         surfaceZ = self.f(finalPosition.x, finalPosition.y)
    #         if surfaceZ is None:
    #             pass
    #         delta = finalPosition.z - surfaceZ

    #     for t in ts:
    #         pt = position + direction * t * maxDistance
    #         surfaceZ = self.z(pt.x, pt.y)
    #         if surfaceZ is not None:
    #             ys.append( pt.z-surfaceZ )
    #             xs.append( t )
        
    # def testAspheric(self):
    #     surface = AsphericSurface(R=1,kappa=0)

    #     for i in range(100):
    #         position = self.randomNegativeZVector()
    #         self.assertTrue(position.z < 0)
    #         final = Vector(0,0,0.5)
    #         d = final-position
    #         distance = d.abs()
    #         direction = d.normalized()

    #         isIntersecting, d, current = surface.intersection(position, direction, distance)
    #         if isIntersecting:
    #             print(current, d)
    #             self.assertAlmostEqual( surface.z(current.x, current.y) - current.z, 0, 4 )
    #         else:
    #             print(i, position, final, d, current)

    def testDomainOfValidityAboveSurface(self):        
        """
        Given a line, I want to know the values of the parameter t that are such that the value of the surface is defined
        Said diffirently, I need the domain of validity for t above the surface f(x,y).

        I did this analytically, because it simply means that the value under the square root must be positive.
        We can replace the x,y from the line into the equation and get an analytical answer.
        I took a picture of my development, it is in this folder.
        """
        M = 10
        I = 1000
        for i in range(I):
            if (i+1) % M == 0:
                print("Iteration: {0}/{1}".format(i+1,I))
                M *= 10

            R = 11
            k = 0.5

            position = self.randomVector()*2*R
            final = self.randomVector()*2*R

            d = final-position
            distance = d.abs()
            direction = d.normalized()

            (u,v,w) = (d.x, d.y, d.z)
            (xo, yo, zo) = (position.x, position.y, position.z)


            # quadratic equation domain validity
            a = u*u+v*v
            b = (2*xo*u + 2*yo*v)
            c = (xo*xo + yo*yo - R*R/(1+k))

            delta = b*b-4*a*c

            # Because of roundoff errors, I need to add a small espilon.
            epsilon = 0.0000001

            validRange = [0,1]

            if delta < 0:
                tMinus = None
                tPlus = None
                validRange = None
            else:
                # tMinus is always smaller than tPlus
                tMinus = (-b-sqrt(delta))/2/a
                tPlus = (-b+sqrt(delta))/2/a
                
                if tMinus > 1 or tPlus < 0:
                    validRange = None
                else:
                    if tMinus > 0:
                        validRange[0] = tMinus+epsilon
                    if tPlus < 1:
                        validRange[1] = tPlus-epsilon

            N = 100
            if validRange is None:
                # Never valid
                for t in linspace(0, 1, N, endpoint=True):
                    pt = self.line(position, d, t)
                    value = self.f(pt.x, pt.y, R=R, kappa=k)
                    self.assertIsNone(value, msg="{0}".format(t))
                continue # Nothing else to validate

            # There is some validity over a finite range
            # Within the range, we should always get a value for f(x,y)
            for t in linspace(tMinus+epsilon, tPlus-epsilon, N, endpoint=True):
                pt = self.line(position, d, t)
                value = self.f(pt.x, pt.y, R=R, kappa=k)
                self.assertIsNotNone(value, msg="{0}".format(t))

            # Our line is really only valid between t=[0,1], so we check
            # the actual limit of the segment, where f(x,y) must be defined
            for t in linspace(validRange[0], validRange[1], N, endpoint=True):
                pt = self.line(position, d, t)
                value = self.f(pt.x, pt.y, R=R, kappa=k)
                self.assertIsNotNone(value, msg="{0}".format(t))

            # Beyond the range, should not be valid
            for t in linspace(tPlus+epsilon, tPlus+1, N, endpoint=True):
                pt = self.line(position, d, t)
                value = self.f(pt.x, pt.y, R=R, kappa=k)
                self.assertIsNone(value, msg="{0}".format(t))

            # Before the range, should not be valid either
            for t in linspace(tMinus-1, tMinus-epsilon, N, endpoint=True):
                pt = self.line(position, d, t)
                value = self.f(pt.x, pt.y, R=R, kappa=k)
                self.assertIsNone(value, msg="{0}".format(t))


    def testDomainOfValidityAboveSurfaceWithSurfaceIMplementation(self):        
        """
        Given a line, I want to know the values of the parameter t that are such that the value of the surface is defined
        Said diffirently, I need the domain of validity for t above the surface f(x,y).

        I did this analytically, because it simply means that the value under the square root must be positive.
        We can replace the x,y from the line into the equation and get an analytical answer.
        I took a picture of my development, it is in this folder.
        """
        M = 10
        I = 10000
        for i in range(I):
            if (i+1) % M == 0:
                print("Iteration: {0}/{1}".format(i+1,I))
                M *= 10

            R = 11
            k = 0.5

            position = self.randomVector()*2*R
            final = self.randomVector()*2*R

            d = final-position
            distance = d.abs()
            direction = d.normalized()

            surface = AsphericSurface(R=R,kappa=k)

            validRange =surface.segmentValidityAboveSurface(position, direction, distance)

            N = 100
            if validRange is None:
                # Never valid
                for t in linspace(0, 1, N, endpoint=True):
                    pt = self.line(position, d, t)
                    value = self.f(pt.x, pt.y, R=R, kappa=k)
                    self.assertIsNone(value, msg="{0}".format(t))
                continue # Nothing else to validate

            # Our line is really only valid between t=[0,1], so we check
            # the actual limit of the segment, where f(x,y) must be defined
            for t in linspace(validRange[0], validRange[1], N, endpoint=True):
                pt = self.line(position, d, t)
                value = self.f(pt.x, pt.y, R=R, kappa=k)
                self.assertIsNotNone(value, msg="{0}".format(t))


if __name__ == '__main__':
    envtest.main()
