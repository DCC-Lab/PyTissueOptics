import numpy as np
import matplotlib.pyplot as plt


class demoPhotons:
    def __init__(self):



class demoStat:
    def __init__(self, min=(-1, -1, 0), max=(1, 1, 2), size=(51, 51, 51)):
        self.min = min
        self.max = max
        self.L = (self.max[0] - self.min[0], self.max[1] - self.min[1], self.max[2] - self.min[2])
        self.size = size
        self.binSizes = (float(self.size[0] - 1) / self.L[0],
                         float(self.size[1] - 1) / self.L[1],
                         float(self.size[2] - 1) / self.L[2])

        self.energy = np.zeros(size)
        self.counter = 0

    def scoreInVolume(self, photons, deltas):
        position = photons.r

        i = np.fromiter(map(int, (self.binSizes[0] * (position.x - self.min[0]) + 0.5)), dtype=np.int32)
        j = np.fromiter(map(int,(self.binSizes[1] * (position.y - self.min[1]) + 0.5)), dtype=np.int32)
        k = np.fromiter(map(int,(self.binSizes[2] * (position.z - self.min[2]) + 0.5)), dtype=np.int32)

         np.where(i < 0)
        if np.any(i < 0 or i > self.size[0] - 1:
            return

        if j < 0 or j > self.size[1] - 1:
            return

        if k < 0 or k > self.size[2] - 1:
            return

        self.energy[i, j, k] += deltas

    def showEnergy2D(self, plane: str, cutAt: int = None, integratedAlong: str = None, title="", realtime=True):
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

        plt.title("Energy in {0} with {1:.0f} photons".format(plane, ))
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

        if realtime:
            self.volumeFig.show()
            plt.pause(0.1)
            plt.clf()
        else:
            plt.ioff()
            self.volumeFig.show()