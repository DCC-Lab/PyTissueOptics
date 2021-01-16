import numpy as np
import matplotlib.pyplot as plt
import time
import json

class Stats:
    def __init__(self, min = (-1, -1, 0), max = (1, 1, 0.5), size = (21,21,21)):
        self.min = min
        self.max = max
        self.L = (self.max[0]-self.min[0],self.max[1]-self.min[1],self.max[2]-self.min[2])
        self.size = size
        self.binSizes = ((self.size[0]-1)/self.L[0],
                         (self.size[1]-1)/self.L[1],
                         (self.size[2]-1)/self.L[2])

        self.photons = set()
        self.savedPhotonCount = 0
        self.energy = np.zeros(size)
        self.figure = None
        self.startTime = time.time()

    @property
    def photonCount(self):
        return len(self.photons)

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

    def report(self):
        elapsed = time.time() - self.startTime
        N = self.photonCount
        print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed/N*1000, N))

    def save(self, filepath="output.json"):
        data = {"min":self.min, "max":self.max, "L":self.L,
                "size":self.size,"energy":self.energy.tolist(),
                "photonCount":len(self.photons)}

        with open(filepath, "w") as write_file:
            json.dump(data, write_file,indent=4, sort_keys=True)

    def restore(self, filepath="output.json"):
        with open(filepath, "r") as read_file:
            data = json.load(read_file)

        self.min = data["min"]
        self.max = data["max"]
        self.L = data["L"]
        self.size = data["size"]
        self.savedPhotonCount = data["photonCount"]
        self.energy = np.array(data["energy"])

    def append(self, filepath="output.json"):
        with open(filepath, "r") as read_file:
            data = json.load(read_file)

        if self.min != data["min"]:
            raise ValueError("To append, data must have same min")
        if self.max != data["max"]:
            raise ValueError("To append, data must have same max")
        if self.L != data["L"]:
            raise ValueError("To append, data must have same L")
        if self.size != data["size"]:
            raise ValueError("To append, data must have same size")

        self.photons.add(set(data["photons"]))
        self.energy = np.add(self.energy, np.array(data["energy"]))

    def score(self, photon, delta):
        self.photons.add(photon.uniqueId)
        position = photon.r

        i = int(self.binSizes[0]*(position.x-self.min[0])-0.5)
        j = int(self.binSizes[1]*(position.y-self.min[1])-0.5)
        k = int(self.binSizes[2]*(position.z-self.min[2])-0.5)

        if i < 0 or i > self.size[0]-1:
            return

        if j < 0 or j > self.size[1]-1:
            return

        if k < 0 or k > self.size[2]-1:
            return

        self.energy[i,j,k] += delta

    def show3D(self):
        raise NotImplementedError()

    def show2D(self, plane:str, cutAt:int= None, integratedAlong:str=None, title="", realtime=True):
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

        plt.title("Energy in {0}, {1} photons".format(plane, len(self.photons)+self.savedPhotonCount))
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
            plt.pause(0.0001)
            plt.clf()
        else:
            plt.ioff()
            plt.show()

    def show1D(self, axis:str, cutAt=None, integratedAlong=None, title="", realtime=True):
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

