import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Qt5Agg')


class CLStatistics:
    def __init__(self, minBounds=(-1, -1, 0), maxBounds=(1, 1, 2), size=(21, 21, 21), opaqueBoundaries=False):
        self.minBounds = minBounds
        self.maxBounds = maxBounds
        self.L = (self.maxBounds[0] - self.minBounds[0], self.maxBounds[1] - self.minBounds[1],
                  self.maxBounds[2] - self.minBounds[2])
        self.size = size
        self.binSizes = (float(self.size[0] - 1) / self.L[0],
                         float(self.size[1] - 1) / self.L[1],
                         float(self.size[2] - 1) / self.L[2])

        self.energy = np.zeros(size)
        self.opaqueBoundaries = opaqueBoundaries
        self.volumeFig = None
        self.volume = []

    @property
    def xCoords(self):
        coords = []
        for i in range(self.size[0]):
            coords.append(self.minBounds[0] + i * (self.maxBounds[0] - self.minBounds[0]) / self.size[0])
        return coords

    @property
    def yCoords(self):
        coords = []
        for i in range(self.size[1]):
            coords.append(self.minBounds[1] + i * (self.maxBounds[1] - self.minBounds[1]) / self.size[1])
        return coords

    @property
    def zCoords(self):
        coords = []
        for i in range(self.size[2]):
            coords.append(self.minBounds[2] + i * (self.maxBounds[2] - self.minBounds[2]) / self.size[2])
        return coords

    @property
    def xBinCenters(self):
        coords = []
        delta = (self.maxBounds[0] - self.minBounds[0]) / self.size[0]
        for i in range(self.size[0]):
            coords.append(self.minBounds[0] + (i + 0.5) * delta)
        return coords

    @property
    def yBinCenters(self):
        coords = []
        delta = (self.maxBounds[1] - self.minBounds[1]) / self.size[1]
        for i in range(self.size[1]):
            coords.append(self.minBounds[1] + (i + 0.5) * delta)
        return coords

    @property
    def zBinCenters(self):
        coords = []
        delta = (self.maxBounds[2] - self.minBounds[2]) / self.size[2]
        for i in range(self.size[2]):
            coords.append(self.minBounds[2] + (i + 0.5) * delta)
        return coords

    def energyRMSVolume(self):
        (xWidth, yWidth, zWidth) = self.energyRMSWidths()
        return xWidth * yWidth * zWidth

    def energyRMSWidths(self):
        xWidth = self.rms(self.xBinCenters, self.energy.sum(axis=(1, 2)))
        yWidth = self.rms(self.yBinCenters, self.energy.sum(axis=(0, 2)))
        zWidth = self.rms(self.zBinCenters, self.energy.sum(axis=(0, 1)))
        return xWidth, yWidth, zWidth

    @staticmethod
    def rms(xs, values):
        vX = 0
        vX2 = 0
        vSum = 0
        for x, value in zip(xs, values):
            vX += value * x
            vX2 += value * x * x
            vSum += value

        xMean = vX / vSum
        x2Mean = vX2 / vSum
        return np.sqrt(x2Mean - xMean * xMean)

    def scoreInVolume(self, position, delta):
        self.volume.append((position, delta))

        i = int(self.binSizes[0] * (position[0] - self.minBounds[0]) + 0.5)
        j = int(self.binSizes[1] * (position[1] - self.minBounds[1]) + 0.5)
        k = int(self.binSizes[2] * (position[2] - self.minBounds[2]) + 0.5)

        if self.opaqueBoundaries:
            if i < 0:
                i = 0
            if i > self.size[0] - 1:
                i = self.size[0] - 1

            if j < 0:
                j = 0
            if j > self.size[1] - 1:
                j = self.size[1] - 1

            if k < 0:
                k = 0
            if k > self.size[2] - 1:
                k = self.size[2] - 1
        else:
            if i < 0 or i > self.size[0] - 1:
                return

            if j < 0 or j > self.size[1] - 1:
                return

            if k < 0 or k > self.size[2] - 1:
                return

        self.energy[i, j, k] += delta

    def showEnergy2D(self, plane: str, cutAt: int = None, integratedAlong: str = None, title=None):
        if len(self.volume) == 0:
            return

        if integratedAlong is None and cutAt is None:
            raise ValueError("You must provide cutAt= or integratedAlong=")
        elif integratedAlong is not None and cutAt is not None:
            raise ValueError("You cannot provide both cutAt= and integratedAlong=")
        elif integratedAlong is None and cutAt is not None:
            if plane == 'xy':
                cutAt = int((self.size[2] - 1) / 2)
            elif plane == 'yz':
                cutAt = int((self.size[0] - 1) / 2)
            elif plane == 'xz':
                cutAt = int((self.size[1] - 1) / 2)

        if self.volumeFig is None:
            plt.ion()
            self.volumeFig = plt.figure()

        if title is None:
            plt.title("Energy in {0} with {1:.0f} photons".format(plane, len(self.volume)))
        elif type(title) == str:
            plt.title(title)

        if cutAt is not None:
            if plane == 'xy':
                plt.imshow(np.log(self.energy[:, :, cutAt] + 0.0001), cmap='viridis',
                           extent=[self.minBounds[0], self.maxBounds[0], self.minBounds[1], self.maxBounds[1]],
                           aspect='auto')
            elif plane == 'yz':
                plt.imshow(np.log(self.energy[cutAt, :, :] + 0.0001), cmap='viridis',
                           extent=[self.minBounds[1], self.maxBounds[1], self.minBounds[2], self.maxBounds[2]],
                           aspect='auto')
            elif plane == 'xz':
                plt.imshow(np.log(self.energy[:, cutAt, :] + 0.0001), cmap='viridis',
                           extent=[self.minBounds[0], self.maxBounds[0], self.minBounds[2], self.maxBounds[2]],
                           aspect='auto')
        else:
            if plane == 'xy':
                energySum = self.energy.sum(axis=2)
                plt.imshow(np.log(energySum + 0.0001), cmap='viridis',
                           extent=[self.minBounds[0], self.maxBounds[0], self.minBounds[1], self.maxBounds[1]],
                           aspect='auto')
            elif plane == 'yz':
                energySum = self.energy.sum(axis=0)
                plt.imshow(np.log(energySum + 0.0001), cmap='viridis',
                           extent=[self.minBounds[1], self.maxBounds[1], self.minBounds[2], self.maxBounds[2]],
                           aspect='auto')
            elif plane == 'xz':
                energySum = self.energy.sum(axis=1)
                plt.imshow(np.log(energySum + 0.0001), cmap='viridis',
                           extent=[self.minBounds[2], self.maxBounds[2], self.minBounds[0], self.maxBounds[0]],
                           aspect='auto')
                plt.pause(20)
