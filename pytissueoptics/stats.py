from pytissueoptics import *
from pytissueoptics.vector import Vector
#import matplotlib.pyplot as plt
import numpy as np


class Stats:
    def __init__(self, globalVolumeStats=True, min=(-1, -1, 0), max=(1, 1, 0.5), size=(21, 21, 21), opaqueBoundaries=False):
        self.min = min
        self.max = max
        self.L = (self.max[0] - self.min[0], self.max[1] - self.min[1], self.max[2] - self.min[2])
        self.size = size
        self.binSizes = (float(self.size[0] - 1) / self.L[0],
                         float(self.size[1] - 1) / self.L[1],
                         float(self.size[2] - 1) / self.L[2])

        self.energy = np.zeros(size)
        self.globalVolumeStats = globalVolumeStats
        self.opaqueBoundaries = opaqueBoundaries
        self.surfaceFig = None
        self.volumeFig = None
        self.volume = []
        self.starting = []
        self.crossing = []
        self.final = []
        self.startTime = time.time()

    @property
    def inputWeight(self):
        total = sum(self.starting)
        if total == 0:
            return 1.0
        return total

    @property
    def photonCount(self):
        return len(self.final)

    @property
    def xCoords(self):
        coords = []
        for i in range(self.size[0]):
            coords.append(self.min[0] + i * (self.max[0] - self.min[0]) / self.size[0])
        return coords

    @property
    def yCoords(self):
        coords = []
        for i in range(self.size[1]):
            coords.append(self.min[1] + i * (self.max[1] - self.min[1]) / self.size[1])
        return coords

    @property
    def zCoords(self):
        coords = []
        for i in range(self.size[2]):
            coords.append(self.min[2] + i * (self.max[2] - self.min[2]) / self.size[2])

        return coords

    @property
    def xBinCenters(self):
        coords = []
        delta = (self.max[0] - self.min[0]) / self.size[0]
        for i in range(self.size[0]):
            coords.append(self.min[0] + (i +0.5) * delta)
        return coords

    @property
    def yBinCenters(self):
        coords = []
        delta = (self.max[1] - self.min[1]) / self.size[1]
        for i in range(self.size[1]):
            coords.append(self.min[1] + (i + 0.5) * delta)
        return coords

    @property
    def zBinCenters(self):
        coords = []
        delta = (self.max[2] - self.min[2]) / self.size[2]
        for i in range(self.size[2]):
            coords.append(self.min[2] + (i + 0.5)* delta)

        return coords

    def energyRMSVolume(self):
        (xWidth, yWidth, zWidth) = self.energyRMSWidths()

        return xWidth*yWidth*zWidth

    def energyRMSWidths(self):
        xWidth = self.rms(self.xBinCenters, self.energy.sum(axis=(1, 2)))
        yWidth = self.rms(self.yBinCenters, self.energy.sum(axis=(0, 2)))
        zWidth = self.rms(self.zBinCenters, self.energy.sum(axis=(0, 1)))

        return (xWidth, yWidth, zWidth)

    def rms(self, xs, values):
        vX  = 0
        vX2 = 0
        vSum = 0
        for x, value in zip(xs, values):
            vX += value*x
            vX2 += value*x*x
            vSum += value

        xMean = vX/vSum
        x2Mean = vX2/vSum

        return np.sqrt(x2Mean-xMean*xMean)

    def photonsCrossingPlane(self, surface, geometryOrigin):
        a = []
        b = []
        weights = []

        for (r, w) in self.crossing:
            isContained, u, v = surface.contains(r - geometryOrigin)

            if isContained:
                a.append(u)
                b.append(v)
                weights.append(w)

        return a, b, weights

    def totalWeightCrossingPlane(self, surface, geometryOrigin) -> float:
        a, b, weights = self.photonsCrossingPlane(surface, geometryOrigin)

        return sum(weights)

    def totalWeightAcrossAllSurfaces(self, surfaces, geometryOrigin) -> float:
        totalWeightAcrossAllSurfaces = 0
        for surface in surfaces:
            totalWeightAcrossAllSurfaces += self.totalWeightCrossingPlane(surface, geometryOrigin)
        return totalWeightAcrossAllSurfaces

    def totalWeightAbsorbed(self) -> float:
        return sum(sum(sum(self.energy)))

    def absorbance(self, referenceWeight=None) -> float:
        if referenceWeight is None:
            referenceWeight = self.inputWeight
        return self.totalWeightAbsorbed() / referenceWeight

    def transmittance(self, surfaces, geometryOrigin, referenceWeight=None) -> float:
        if referenceWeight is None:
            referenceWeight = self.inputWeight
        return self.totalWeightAcrossAllSurfaces(surfaces, geometryOrigin) / referenceWeight

    def report(self):
        elapsed = time.time() - self.startTime
        N = self.photonCount
        print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed / N * 1000, N))

    def scoreInVolume(self, photon, delta):
        if self.globalVolumeStats:
            self.volume.append((Vector(photon.globalPosition), delta))
            position = photon.globalPosition
        else:
            self.volume.append((Vector(photon.r), delta))
            position = photon.r

        i = int(self.binSizes[0] * (position.x - self.min[0]) + 0.5)
        j = int(self.binSizes[1] * (position.y - self.min[1]) + 0.5)
        k = int(self.binSizes[2] * (position.z - self.min[2]) + 0.5)

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

            if k < 0 or  k > self.size[2] - 1:
                return

        self.energy[i, j, k] += delta

    def scoreWhenCrossing(self, photon):
        self.crossing.append((Vector(photon.globalPosition), photon.weight))

    def scoreWhenStarting(self, photon):
        self.starting.append(photon.weight)

    def scoreWhenFinal(self, photon):
        self.final.append(photon)

    def showEnergy3D(self):
        raise NotImplementedError()

    def showEnergy2D(self, plane: str, cutAt: int = None, integratedAlong: str = None, title=None, xLabel=None, yLabel=None, realtime=True):
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
            plt.title("Energy in {0} with {1:.0f} photons".format(plane, self.inputWeight))
        elif type(title) == str:
            plt.title(title)

        if cutAt is not None:
            if plane == 'xy':
                plt.imshow(np.log(self.energy[:, :, cutAt] + 0.0001), cmap='viridis',
                           extent=[self.min[0], self.max[0], self.min[1], self.max[1]], aspect='auto')
            elif plane == 'yz':
                plt.imshow(np.log(self.energy[cutAt, :, :] + 0.0001), cmap='viridis',
                           extent=[self.min[1], self.max[1], self.min[2], self.max[2]], aspect='auto')
            elif plane == 'xz':
                plt.imshow(np.log(self.energy[:, cutAt, :] + 0.0001), cmap='viridis',
                           extent=[self.min[0], self.max[0], self.min[2], self.max[2]], aspect='auto')
        else:
            if plane == 'xy':
                sum = self.energy.sum(axis=2)
                plt.imshow(np.log(sum + 0.0001), cmap='viridis',
                           extent=[self.min[0], self.max[0], self.min[1], self.max[1]], aspect='auto')
            elif plane == 'yz':
                sum = self.energy.sum(axis=0)
                plt.imshow(np.log(sum + 0.0001), cmap='viridis',
                           extent=[self.min[1], self.max[1], self.min[2], self.max[2]], aspect='auto')
            elif plane == 'xz':
                sum = self.energy.sum(axis=1)
                plt.imshow(np.log(sum + 0.0001), cmap='viridis',
                           extent=[self.min[2], self.max[2], self.min[0], self.max[0]], aspect='auto')
                plt.pause(1)

        if realtime:
            #self.volumeFig.show()
            plt.pause(0.1)
            plt.clf()
        else:
            plt.ioff()
            #self.volumeFig.show()

    def showEnergy1D(self, axis: str, cutAt=None, integratedAlong=None, title="", realtime=True):
        if len(self.volume) == 0:
            return

        if integratedAlong is None and cutAt is None:
            # Assume integral
            raise ValueError("You should provide cutAt=(x0, x1) or integratedAlong='xy'.")
        elif integratedAlong is not None and cutAt is not None:
            raise ValueError("You cannot provide both cutAt= and integratedAlong=")
        elif integratedAlong is None and cutAt is not None:
            if axis == 'x':
                cutAt = (int((self.size[1] - 1) / 2), int((self.size[2] - 1) / 2))
            elif axis == 'y':
                cutAt = (int((self.size[0] - 1) / 2), int((self.size[2] - 1) / 2))
            elif axis == 'z':
                cutAt = (int((self.size[0] - 1) / 2), int((self.size[1] - 1) / 2))

        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()

        plt.title(title)
        if cutAt is not None:
            if axis == 'z':
                plt.plot(self.zCoords, np.log10(self.energy[cutAt[0], cutAt[1], :] + 0.0001), 'ko--')
            elif axis == 'y':
                plt.plot(self.yCoords, np.log10(self.energy[cutAt[0], :, cutAt[1]] + 0.0001), 'ko--')
            elif axis == 'x':
                plt.plot(self.xCoords, np.log10(self.energy[:, cutAt[0], cutAt[1]] + 0.0001), 'ko--')
        else:
            if axis == 'z':
                sum = self.energy.sum(axis=(0, 1))
                plt.plot(self.zCoords, np.log10(sum + 0.0001), 'ko--')
            elif axis == 'y':
                sum = self.energy.sum(axis=(0, 2))
                plt.plot(self.yCoords, np.log10(sum + 0.0001), 'ko--')
            elif axis == 'x':
                sum = self.energy.sum(axis=(1, 2))
                plt.plot(self.xCoords, np.log10(sum + 0.0001), 'ko--')

        if realtime:
            plt.show()
            plt.pause(0.0001)
            plt.clf()
        else:
            plt.ioff()
            plt.show()

    def showSurfaceIntensities(self, surfaces, maxPhotons, geometryOrigin, bins=21):
        if len(self.crossing) == 0:
            return

        self.surfaceFig, axes = plt.subplots(nrows=2, ncols=max(1, len(surfaces) // 2), figsize=(14, 8))
        N = maxPhotons

        for i, surface in enumerate(surfaces):
            a, b, weights = self.photonsCrossingPlane(surface, geometryOrigin)
            if len(surfaces) > 2:
                axes[i % 2, i // 2].set_title('Intensity at {0} [T={1:.1f}%]'.format(surface, 100 * sum(weights) / N))
                axes[i % 2, i // 2].hist2d(a, b, weights=weights, bins=bins)
            elif len(surfaces) == 1:
                axes[0].set_title('Intensity at {0} [T={1:.1f}%]'.format(surface, 100 * sum(weights) / N))
                axes[0].hist2d(a, b, weights=weights, bins=bins)
            else:
                self.surfaceFig.set_size_inches(4, 8)
                axes[i % 2].set_title('Intensity at {0} [T={1:.1f}%]'.format(surface, 100 * sum(weights) / N))
                axes[i % 2].hist2d(a, b, weights=weights, bins=bins)

        plt.ioff()
        plt.show()

    @property
    def hasDisplay(self):
        try:
            os.environ['DISPLAY']
        except Exception:
            return False

        return True
