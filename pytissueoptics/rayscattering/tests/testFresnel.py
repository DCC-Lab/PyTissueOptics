import unittest


class TestFresnelRefractionFactory(unittest.TestCase):
    def testWhenRefractWithPerpendicularSurface_shouldNotRotatePhoton(self):
        self.fail()
        # initialDirection = self.DIRECTION.copy()
        # n1, n2 = 1, 1.4
        # surfaceElement = Polygon([Vector()], normal=Vector(0, 0, 1),
        #                          insideMaterial=Material(index=n2),
        #                          outsideMaterial=Material(index=n1))
        # intersection = Intersection(1, self.POSITION + self.DIRECTION, surfaceElement)
        #
        # fresnelIntersection = FresnelIntersect(self.photon.direction, intersection)
        # self.photon.refract(fresnelIntersection)
        #
        # self.assertEqual(initialDirection, self.photon.direction)

    def testWhenRefract_shouldOrientPhotonTowardsFresnelRefractionAngle(self):
        self.fail()
        # incidenceAngle = math.pi / 10
        # self.photon = Photon(self.POSITION, Vector(0, math.sin(incidenceAngle), -math.cos(incidenceAngle)))
        # n1, n2 = 1, 1.4
        # surfaceElement = Polygon([Vector()], normal=Vector(0, 0, 1),
        #                          insideMaterial=Material(index=n2),
        #                          outsideMaterial=Material(index=n1))
        # intersection = Intersection(1, self.POSITION + self.DIRECTION, surfaceElement)
        #
        # fresnelIntersection = FresnelIntersection(self.photon.direction, intersection)
        # self.photon.refract(fresnelIntersection)
        #
        # refractionAngle = math.asin(n1/n2 * math.sin(incidenceAngle))
        # expectedDirection = Vector(0, math.sin(refractionAngle), -math.cos(refractionAngle))
        # self.assertVectorEqual(expectedDirection, self.photon.direction)
