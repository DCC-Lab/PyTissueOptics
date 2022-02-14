class Rotation:
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

    def add(self, other: 'Rotation'):
        self._xTheta += other.xTheta
        self._yTheta += other.yTheta
        self._zTheta += other.zTheta

    @property
    def components(self) -> tuple:
        return self._xTheta, self._yTheta, self._zTheta

    def getInverse(self) -> 'Rotation':
        return Rotation(xTheta=-self._xTheta, yTheta=-self._yTheta, zTheta=-self._zTheta)

    def __bool__(self):
        return self._xTheta != 0 or self._yTheta != 0 or self._zTheta != 0
