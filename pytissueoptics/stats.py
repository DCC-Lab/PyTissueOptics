import numpy as np
import matplotlib.pyplot as plt
import time
import os
from .vector import *

class Stats:
    def __init__(self, min = (-1, -1, 0), max = (1, 1, 0.5), size = (21,21,21)):
        self.min = min
        self.max = max
        self.L = (self.max[0]-self.min[0],self.max[1]-self.min[1],self.max[2]-self.min[2])
        self.size = size
        self.binSizes = ((self.size[0]-1)/self.L[0],
                         (self.size[1]-1)/self.L[1],
                         (self.size[2]-1)/self.L[2])

        self.energy = np.zeros(size)
        self.figure = None
        self.volume = None
        self.crossing = []
        self.final = []
        self.startTime = time.time()

    @property
    def photonCount(self):
        return len(self.final)

    @property
    def xCoords(self):
        coords = []
        for i in range(self.size[0]):
            coords.append(self.min[0] + i * (self.max[0]-self.min[0])/self.size[0])
        return coords

    @property
    def yCoords(self):
        coords = []
        for i in range(self.size[1]):
            coords.append(self.min[1] + i * (self.max[1]-self.min[1])/self.size[1])
        return coords

    @property
    def zCoords(self):
        coords = []
        for i in range(self.size[2]):
            coords.append(self.min[2] + i * (self.max[2]-self.min[2])/self.size[2])

        return coords

    def photonsCrossingPlane(self, surface):
        a = []
        b = []
        weights = []

        for (r, w) in self.crossing:
            isContained, u, v = surface.contains(r)
            if isContained:
                a.append(u)
                b.append(v)
                weights.append(w)

        return a, b, weights

    def totalWeightCrossingPlane(self, surface) -> float:
        a, b, weights = self.photonsCrossingPlane(surface)

        return sum(weights)

    def totalWeightAbsorbed(self) -> float:
        return sum(sum(sum(self.energy)))

    def report(self):
        elapsed = time.time() - self.startTime
        N = self.photonCount
        print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed/N*1000, N))

    def scoreInVolume(self, photon, delta):
        position = photon.r

        i = int(self.binSizes[0]*(position.x-self.min[0])-0.5)
        j = int(self.binSizes[1]*(position.y-self.min[1])-0.5)
        k = int(self.binSizes[2]*(position.z-self.min[2])-0.5)

        if i < 0: 
            i = 0
        if i > self.size[0]-1:
            i = self.size[0]-1
        
        if j < 0:
            j = 0
        if j > self.size[1]-1:
            j = self.size[1]-1

        if k < 0:
            k = 0
        if k > self.size[2]-1:
            k = self.size[2]-1

        self.energy[i,j,k] += delta

    def scoreWhenCrossing(self, photon, surface):
        self.crossing.append( (Vector(photon.r), photon.weight) )

    def scoreWhenFinal(self, photon):
        self.final.append(photon)

    def showEnergy3D(self):
        raise NotImplementedError()

    def showEnergy2D(self, plane:str, cutAt:int= None, integratedAlong:str=None, title="", realtime=True):
        if integratedAlong is None and cutAt is None:
            raise ValueError("You must provide cutAt= or integratedAlong=")
        elif integratedAlong is not None and cutAt is not None:
            raise ValueError("You cannot provide both cutAt= and integratedAlong=")
        elif integratedAlong is None and cutAt is not None:
            if plane == 'xy':
                cutAt = int((self.size[2]-1)/2)
            elif plane == 'yz':
                cutAt = int((self.size[0]-1)/2)
            elif plane == 'xz':
                cutAt = int((self.size[1]-1)/2)

        if self.figure == None:
            plt.ion()
            self.figure = plt.figure()

        plt.title("Energy in {0}, {1} photons".format(plane, self.photonCount))
        if cutAt is not None:
            if plane == 'xy':
                plt.imshow(np.log(self.energy[:,:,cutAt]+0.0001),cmap='hsv',extent=[self.min[0],self.max[0],self.min[1],self.max[1]],aspect='auto')
            elif plane == 'yz':
                plt.imshow(np.log(self.energy[cutAt,:,:]+0.0001),cmap='hsv',extent=[self.min[1],self.max[1],self.min[2],self.max[2]],aspect='auto')
            elif plane == 'xz':
                plt.imshow(np.log(self.energy[:,cutAt,:]+0.0001),cmap='hsv',extent=[self.min[0],self.max[0],self.min[2],self.max[2]],aspect='auto')
        else:
            if plane == 'xy':
                sum = self.energy.sum(axis=2)
                plt.imshow(np.log(sum+0.0001),cmap='hsv',extent=[self.min[0],self.max[0],self.min[1],self.max[1]],aspect='auto')
            elif plane == 'yz':
                sum = self.energy.sum(axis=0)
                plt.imshow(np.log(sum+0.0001),cmap='hsv',extent=[self.min[1],self.max[1],self.min[2],self.max[2]],aspect='auto')
            elif plane == 'xz':
                sum = self.energy.sum(axis=1)
                plt.imshow(np.log(sum+0.0001),cmap='hsv',extent=[self.min[2],self.max[2],self.min[0],self.max[0]],aspect='auto')

        if realtime:
            plt.show()
            plt.pause(0.1)
            plt.clf()
        else:
            plt.ioff()
            plt.show()

    def showEnergy1D(self, axis:str, cutAt=None, integratedAlong=None, title="", realtime=True):
        if integratedAlong is None and cutAt is None:
            # Assume integral
            raise ValueError("You should provide cutAt=(x0, x1) or integratedAlong='xy'.")
        elif integratedAlong is not None and cutAt is not None:
            raise ValueError("You cannot provide both cutAt= and integratedAlong=")
        elif integratedAlong is None and cutAt is not None:
            if axis == 'x':
                cutAt = (int((self.size[1]-1)/2),int((self.size[2]-1)/2))
            elif axis == 'y':
                cutAt = (int((self.size[0]-1)/2),int((self.size[2]-1)/2))
            elif axis == 'z':
                cutAt = (int((self.size[0]-1)/2),int((self.size[1]-1)/2))

        if self.figure == None:
            plt.ion()
            self.figure = plt.figure()

        plt.title(title)
        if cutAt is not None:
            if axis == 'z':
                plt.plot(self.zCoords, np.log10(self.energy[cutAt[0],cutAt[1],:]+0.0001),'ko--')
            elif axis == 'y':
                plt.plot(self.yCoords, np.log10(self.energy[cutAt[0],:,cutAt[1]]+0.0001),'ko--')
            elif axis == 'x':
                plt.plot(self.xCoords, np.log10(self.energy[:,cutAt[0],cutAt[1]]+0.0001),'ko--')
        else:
            if axis == 'z':
                sum = self.energy.sum(axis=(0,1))
                plt.plot(self.zCoords, np.log10(sum+0.0001),'ko--')
            elif axis == 'y':
                sum = self.energy.sum(axis=(0,2))
                plt.plot(self.yCoords, np.log10(sum+0.0001),'ko--')
            elif axis == 'x':
                sum = self.energy.sum(axis=(1,2))
                plt.plot(self.xCoords, np.log10(sum+0.0001),'ko--')

        if realtime:
            plt.show()
            plt.pause(0.0001)
            plt.clf()
        else:
            plt.ioff()
            plt.show()

    def showSurfaceIntensities(self, surfaces):
        fig, axes = plt.subplots(nrows=2, ncols=len(surfaces)//2, figsize=(14,8))
        N = len(self.final)

        for i, surface in enumerate(surfaces):
            a,b,weights = self.photonsCrossingPlane(surface)
            if len(surfaces) > 2:
                axes[i % 2, i // 2].set_title('Intensity at {0} [T={1:.1f}%]'.format(surface,100*sum(weights)/N))
                axes[i % 2, i // 2].hist2d(a,b,weights=weights, bins=21)
            else:
                fig.set_size_inches(4,8)
                axes[i % 2].set_title('Intensity at {0} [T={1:.1f}%]'.format(surface,100*sum(weights)/N))
                axes[i % 2].hist2d(a,b,weights=weights, bins=21)

        fig.tight_layout()
        plt.ioff()
        plt.show()

    @property
    def hasDisplay(self):
        try:
            os.environ['DISPLAY']
        except:
            return False

        return True
