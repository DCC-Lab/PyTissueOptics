from enum import Enum


class Direction(Enum):
    X_POS = 0
    Y_POS = 1
    Z_POS = 2
    X_NEG = 3
    Y_NEG = 4
    Z_NEG = 5

    def isSameAxisAs(self, other) -> bool:
        return self.value % 3 == other.value % 3

    @property
    def axis(self) -> int:
        """ Returns an integer between 0 and 2 representing the x, y, or z axis, ignoring direction sign. """
        return self.value % 3

    @property
    def isNegative(self) -> bool:
        return self.value >= 3

    @property
    def isPositive(self) -> bool:
        return not self.isNegative

    @property
    def sign(self) -> int:
        return 1 if self.isPositive else -1


DEFAULT_X_VIEW_DIRECTIONS = (Direction.X_POS, Direction.Z_POS)
DEFAULT_Y_VIEW_DIRECTIONS = (Direction.Y_NEG, Direction.Z_POS)
DEFAULT_Z_VIEW_DIRECTIONS = (Direction.Z_POS, Direction.X_NEG)
