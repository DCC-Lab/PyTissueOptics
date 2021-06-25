import envtest  # modifies path
from pytissueoptics import *
from numpy import linspace, pi, sqrt, polyfit
from pytissueoptics.surface import AsphericSurface
import matplotlib.pyplot as plt

inf = float("+inf")

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

    def f(self, x, y, R=35, diameter=25):
        try:
            r2 = x*x+y*y
            if r2 > diameter*diameter:
                return None
            value = np.sqrt(R*R-r2)
            if isnan(value):
                return None
            return value
        except:
            return None

    def c(self, x,y, R, kappa=0):
        r = np.sqrt(x*x+y*y)
        z = r*r/(R*(1+sqrt(1-(1+kappa)*r*r/R/R)))   

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

            print(position, current, wasBelow, isBelow, t, maxDistance)
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
if __name__ == '__main__':
    envtest.main()
