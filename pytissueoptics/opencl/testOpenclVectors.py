import pyopencl as pycl
import numpy as np
import ctypes
from collections import namedtuple


N = 10000

BufferTuple = namedtuple("BufferTuple", ["objectId", "bufferObject", "objectShape"])

class GPUManager:
    def __init__(self):
        self.platforms = pycl.get_platforms()
        self.devices = self.platforms[0].get_devices(pycl.device_type.GPU)
        print(self.platforms, self.devices)
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
        self.bufferTuples = []

    def createReadWriteMemoryBuffer(self, array):
        return pycl.Buffer(self.context, flags=pycl.mem_flags.READ_WRITE, size=array.nbytes)

    def copyBufferToGPUDevice(self, array, buffer):
        pycl.enqueue_copy(self.queue, src=array, dest=buffer)
        self.queue.finish()

    def copyBufferToHostDevice(self, arrayObject, buffer):
        pycl.enqueue_copy(self.queue, src=buffer, dest=arrayObject)
        self.queue.finish()

    def isInBufferMemory(self, objectId):
        if objectId in self.bufferTuples:
            print("id is in buffer memory")
            self.getBufferTupleContaining(objectId)
        else:
            try:
                array = ctypes.cast(objectId, ctypes.py_object).value.v
                buffer = self.createReadWriteMemoryBuffer(array)
                self.copyBufferToGPUDevice(array, buffer)
                bufferTuple = BufferTuple(objectId, buffer, array.shape)
                self.bufferTuples.append(bufferTuple)
                return bufferTuple

            except Exception as e:
                print("An error occured during buffer handling.")

        print("you are here.")

    def getBufferTupleContaining(self, item):
        for i, tuple in enumerate(self.bufferTuples):
            if item in tuple:
                return tuple
            else:
                continue

    def createEmptyLikeBufferWithId(self, objectShape, id):
        emptyArray = np.empty_like(objectShape)
        buffer = self.createReadWriteMemoryBuffer(emptyArray)
        self.copyBufferToGPUDevice(emptyArray, buffer)
        self.bufferTuples.append(BufferTuple(id, buffer, objectShape))
        return buffer, objectShape

    def getBufferContaining(self, item):
        tuple = self.getBufferTupleContaining(item)
        return tuple.bufferObject

    def add(self, id_1, id_2, id_result):
        bufferTuple1 = self.isInBufferMemory(id_1)
        bufferTuple2 = self.isInBufferMemory(id_2)
        resultBuffer, arrayShape = self.createEmptyLikeBufferWithId(bufferTuple1.objectShape, id_result)
        self.program.sum(self.queue, (9, 10), (9, 1), *[bufferTuple1.bufferObject, bufferTuple2.bufferObject, resultBuffer])
        self.queue.finish()
        return

    def copyToHostArray(self, id):
        tuple = self.getBufferTupleContaining(id)
        array = ctypes.cast(id, ctypes.py_object).value.v
        self.copyBufferToHostDevice(array, tuple.bufferObject)


class OpenclVectors:
    def __init__(self, vectors=None, N=None, vectorsId=None):
        self.set_v(vectors, N)
        if vectorsId is not None:
            self.id = vectorsId
        else:
            self.id = id(self)

        self._iteration = 0

    def set_v(self, vectors=None, N=None):
        if vectors is not None:
            if type(vectors) == np.ndarray:
                self.v = vectors.astype('float32')

            else:
                self.v = np.asarray(vectors, dtype=np.float64)
        elif N is not None:
            self.v = np.zeros((N, 3), dtype=np.float64)

        else:
            self.v = None

    def __add__(self, other):
        resultObject = OpenclVectors()
        GPUManager().add(id(self), id(other), id(resultObject))
        return resultObject

    def __repr__(self):
        print(str(self.v))

    def update(self):
        GPUManager().copyToHostArray(id(self))



a = OpenclVectors(np.ones(N))
b = a + a
b.update()
print(b)


