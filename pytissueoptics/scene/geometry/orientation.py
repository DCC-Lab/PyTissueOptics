class Orientation:
    def __init__(self, xTheta: float = 0, yTheta: float = 0, zTheta: float = 0):
        self._xTheta = xTheta
        self._yTheta = yTheta
        self._zTheta = zTheta

    @property
    def xTheta(self):
        return self._xTheta

    @property
    def yTheta(self):
        return self._yTheta

    @property
    def zTheta(self):
        return self._zTheta

    def add(self, other: 'Orientation'):
        self._xTheta += other.xTheta
        self._yTheta += other.yTheta
        self._zTheta += other.zTheta
