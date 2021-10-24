import pyopencl as pycl
import numpy as np
import ctypes


N = 10000

class GPUManager:
    def __init__(self):
        if GPUManager.instance is None:
            GPUManager.instance = self

        self.platforms = [platform for platform in pycl.get_platforms()]
        self.devices = self.platforms[0].get_devices()
        self.context = pycl.Context(devices=self.devices)
        self.queue = pycl.CommandQueue(self.context)
        programSource = """
              kernel void sum(global float *a, 
                              global float *b,
                              global float *c){
                int gid = get_global_id(0);
                c[gid] = a[gid] + b[gid];
              }

                kernel void multiply(global float *a, 
                              global float *b,
                              global float *c){
                int gid = get_global_id(0);
                c[gid] = a[gid] * b[gid];
              }
            """
        programSource = pycl.Program(self.context, programSource)
        self.program = programSource.build()
        self.ids = []

    def createReadWriteMemoryBuffer(self, array):
        return pycl.Buffer(self.context,flags=pycl.mem_flags.READ_WRITE,size=array.nbytes)

    def copyBufferToGPUDevice(self, array, buffer):
        pycl.enqueue_copy(self.queue, src=array, dest=buffer)
        self.queue.finish()

    def copyBufferToHostDevice(self, id):
        pass

    def isInBufferMemory(self, objectId):
        if objectId in self.ids:
            print("id is in buffer memory")
        else:
            try:
                array = ctypes.cast(objectId, ctypes.py_object).value.v
                buffer = self.createReadWriteMemoryBuffer(array)
                self.copyBufferToGPUDevice(array, buffer)
                self.ids.append((objectId, id(buffer), buffer))

            except Exception as e:
                print("An error occured during buffer handling.")

        print("")

    def createResultBuffer

    def getBuffer(self, id):


    def add(self, id_1, id_2):
        self.isInBufferMemory(id_1)
        self.isInBufferMemory(id_2)
        self.createResultBuffer()



class OpenclVectors:
    def __init__(self, vectors=None, N=None, vectorsId=None):
        if vectors is not None:
            if type(vectors) == np.ndarray:
                self.v = vectors.astype('float32')

            else:
                self.v = np.asarray(vectors, dtype=np.float64)
        elif N is not None:
            self.v = np.zeros((N, 3), dtype=np.float64)

        self._iteration = 0

        if vectorsId is not None:
            self.id = vectorsId
        else:
            self.id = id(self)

        # find the instance of GPUManager on its own
        # self.GPUManager = findGPUManager()

    def __add__(self, other):
        resultId = GPUManager().add(id(self), id(other))
        return OpenclVectors(vectorsId=resultId)
        # send the 2 object (self and other) object ID to the GPU Manager and send the add function.

    def stdOut(self):
        GPUManager().copyToHost(id(self))

a = OpenclVectors(np.ones(N))
b = a+a


