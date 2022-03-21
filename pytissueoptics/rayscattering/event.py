from typing import NamedTuple
from ctypes import *
# from photon import Photon

"""
Photon events

MOVE 
MOVEBY, distance
SCATTER, µs, µa, g
SCATTERBY, theta, phi
REFLECT normal, n1, n2 
REFRACT normal, n1, n2

Geometry
GET_INTERSECTION

Physics
GET_SCATTERING_ANGLES
GET_SCATTERING_DISTANCE
GET_IS_REFLECTED 
"""

from enum import IntEnum

class Vector:
    pass

class Name(IntEnum):
    MOVE = 1
    MOVEBY = 9
    SCATTER = 2
    SCATTERBY = 10
    AT_INTERFACE = 3
    REFLECT = 4
    REFRACT = 5
    ROULETTE = 6
    STORE = 11

class CompactMaterial(NamedTuple):
    mu_s:float = 0
    mu_a:float = 0
    g:float = 0
    n1:float = 0
    n2:float = 0

class Event(NamedTuple):
    which: Name
    processing : None
    distance: float = None
    theta: float = None
    phi: float = None
    intersection:Intersection = None
    materialInside:CompactMaterial = None
    materialOutside:CompactMaterial = None


class EventDrivenPhoton(Photon):
    def __init__(self, position, direction):
        self.nextEvent = None
        super().__init__(position, direction)

    def processEvent(self):
        while self.nextEvent is not None:
            self.nextEvent.processing()

    def processMoveEvent(self):
        d = self._material.getScatteringDistance()
        self.nextEvent = Event(MOVEBY, self.processMoveByEvent, distance=d)

    def processMoveByEvent(self):
        distance = self.nextEvent.distance
        self._getIntersection(distance)

        if intersection is None:
            self.moveBy(distance)
            self.nextEvent = Event(SCATTER, self.processScatterEvent)
        else:
            self.moveBy(intersection.distance)
            fresnelIntersection = FresnelIntersect(self._direction, intersection)
            self.nextEvent = Event(REFLECT_OR_REFRACT, self.reflectOrRefractAtInterfaceEvent, fresnelIntersection)

    def reflectOrRefractAtInterfaceEvent(self):
        interfaceEvent = self.nextEvent
        isReflected = interfaceEvent.fresnelIntersection.isReflected()
        if isReflected:
            self.nextEvent = Event(REFLECT, self.processReflectEvent, fresnelIntersection)
        else:
            self.nextEvent = Event(REFRACT, self.processRefractEvent, fresnelIntersection)

    def processScatterEvent(self):
        theta, phi = self._material.getScatteringAngles()
        self.nextEvent = Event(SCATTERBY, self.processRouletteEvent)

    def processScatterByEvent(self):
        self.scatter(self.nextEvent.theta, self.nextEvent.phi)
        self.decreaseWeight(0.1)
        self.nextEvent = Event(ROULETTE, self.processRouletteEvent)

    def processReflectEvent(self):
        interfaceEvent = self.nextEvent
        self.reflect(interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)
        self.nextEvent = Event(MOVEBY, self.processMoveEvent, remainingDistance)

    def processRefractEvent(self):
        interfaceEvent = self.nextEvent
        self.refract(interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)
        self.nextEvent = Event(MOVEBY, self.processMoveEvent, remainingDistance)

    def processRouletteEvent(self):
        isDead =  self._roulette()
        if isDead:
            self.nextEvent = None
        else:
            self.weight *= 10            
            self.nextEvent = Event(MOVE, self.processMoveEvent)


