from enum import Enum, auto


class EnergyType(Enum):
    """
    Type of volumetric energy: either as the deposited energy in the solid (absorption) or as the fluence rate.
    """

    DEPOSITION = auto()
    FLUENCE_RATE = auto()
