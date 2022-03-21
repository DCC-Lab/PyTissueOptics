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


class EventDrivenPhoton:
    def __init__(self, position, direction):
        self.nextEvent = None
        super().__init__(position, direction)

    def processEvent(self):
        while self.nextEvent is not None:
            self.nextEvent.processing()

    def processMoveEvent(self):
        d = self.getScatteringDistance(mu_s, mu_a, g)
        self.nextEvent = Event(MOVEBY, self.processMoveByEvent, distance=d)

    def processMoveByEvent(self):
        intersection = self.getIntersection(d)
        if intersection is None:
            self.move(event.distance)
            self.nextEvent = Event(SCATTER, self.processScatterEvent)
        else:
            self.move(intersection.distance)
            self.nextEvent = Event(AT_INTERFACE, self.processAtInterfaceEvent, intersection.normal, intersection.n1, intersection.n2)

    def processAtInterfaceEvent(self):
        interfaceEvent = self.nextEvent
        isReflected = self.isReflected(interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)
        if isReflected:
            self.nextEvent = Event(REFLECT, self.processReflectEvent, interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)
        else:
            self.nextEvent = Event(REFRACT, self.processRefractEvent, interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)

    def processScatterEvent(self):
        theta, phi = self.getScatteringAngles(mu_s, mu_a, g)
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
        isDead =  self.roulette()
        if isDead:
            self.nextEvent = None
        else:
            self.weight *= 10            
            self.nextEvent = Event(MOVE, self.processMoveEvent)


