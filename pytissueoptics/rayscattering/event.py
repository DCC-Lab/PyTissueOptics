from typing import NamedTuple
# from photon import Photon

"""
Photon events

MOVE, distance
SCATTER, µs, µa
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
    SCATTER = 2
    AT_INTERFACE = 3
    REFLECT = 4
    REFRACT = 5
    ROULETTE = 6

class Event(NamedTuple):
    which: Name
    processing : None
    distance: float = None
    n1: float = None
    n2: float = None
    normal: Vector = None
    material1 = None
    material2 = None


class EventDrivenPhoton:
    # MOVE = Event(Name.MOVE, EventDrivenPhoton.processMoveEvent)
    # SCATTER = Event(Name.SCATTER, EventDrivenPhoton.processScatterEvent)
    # AT_INTERFACE = Event(Name.AT_INTERFACE, EventDrivenPhoton.processAtInterfaceEvent)
    # REFLECT = Event(Name.REFLECT, EventDrivenPhoton.processReflectEvent)
    # REFRACT = Event(Name.REFRACT, EventDrivenPhoton.processRefractEvent)

    def __init__(self, position, direction):
        self.nextEvent = None
        super().__init__(position, direction)

    def processEvent(self):
        while self.nextEvent is not None:
            self.nextEvent.processing()

    def processMoveEvent(self):
        intersection = self.getIntersection(event.distance)
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
        self.scatter(theta, phi)
        d = self.getScatteringDistance(mu_s, mu_a, g)
        self.nextEvent = Event(ROULETTE, self.processRouletteEvent)

    def processReflectEvent(self):
        interfaceEvent = self.nextEvent
        self.reflect(interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)
        self.nextEvent = Event(MOVE, self.processMoveEvent)

    def processRefractEvent(self):
        interfaceEvent = self.nextEvent
        self.refract(interfaceEvent.normal, interfaceEvent.n1, interfaceEvent.n2)
        self.nextEvent = Event(MOVE, self.processMoveEvent)

    def processRouletteEvent(self):
        isDead =  self.roulette()
        if isDead:
            self.nextEvent = None
        else:
            self.weight *= 10            
            self.nextEvent = Event(MOVE, self.processMoveEvent)


