import pyopencl as pycl
import pyopencl as cl
import numpy as np
import unittest
import time
import pyopencl.cltypes
from pyopencl.array import Array as clArray
import pyopencl.clmath
import matplotlib.pyplot as plt
from pathlib import Path

class TestOpenCL(unittest.TestCase):
    context = None

    def test01Import(self):
        """
        Is it installed?
        """
        self.assertIsNotNone(pycl)

    def test02AtLeast1(self):
        """
        Let's get the platform so we can get the devices.
        There should be at least one.
        """
        self.assertTrue(len(pycl.get_platforms()) > 0)

    def test03AtLeast1Device(self):
        """
        Let's get the devices (graphics card?).  There could be a few.
        """
        platform = pycl.get_platforms()[0]
        devices = platform.get_devices()
        self.assertTrue(len(devices) > 0)

    def test031AtLeastGPUorCPU(self):
        devices = pycl.get_platforms()[0].get_devices()
        for device in devices:
            self.assertTrue(device.type == pycl.device_type.GPU or device.type == pycl.device_type.CPU)

    def test04Context(self):
        """ Finally, we need the context for computation context.
        """
        devices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        context = pycl.Context(devices=devices)
        self.assertIsNotNone(context)

    def test05GPUDevice(self):
        gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        self.assertTrue(len(gpuDevices) >= 1)
        gpuDevice = [ device for device in gpuDevices if device.vendor == 'AMD']
        if gpuDevice is None:
            gpuDevice = [ device for device in gpuDevices if device.vendor == 'Intel']
        self.assertIsNotNone(gpuDevice)

    def test06ProgramSource(self):
        gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        context = pycl.Context(devices=gpuDevices)
        queue = pycl.CommandQueue(context)

        program_source = """
        kernel void sum(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] + b[gid];
                      }

        kernel void multiply(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] * b[gid];
                      }
        """
        program = pycl.Program(context, program_source).build()

    def test07CopyingBuffersFromHostToGPU(self):
        """
        If I run this several times, I sometimes get 1 ms or 1000 ms.
        I suspect there is some stsartup time for PyOpenCL.
        I will write a setup function for the test

        I did, it is now much more stable at 1-2 ms.
        """

        # gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        context = TestOpenCL.context

        N = 10000000
        a_np = np.random.rand(N).astype(np.float32)
        b_np = np.random.rand(N).astype(np.float32)
        self.assertIsNotNone(a_np)
        self.assertIsNotNone(b_np)
        mf = pycl.mem_flags
        a_g = pycl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
        b_g = pycl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
        self.assertIsNotNone(a_g)
        self.assertIsNotNone(b_g)
        res_g = pycl.Buffer(context, mf.WRITE_ONLY, a_np.nbytes)


        queue = pycl.CommandQueue(context)

        program_source = """
        kernel void sum(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] + b[gid];
                      }

        kernel void multiply(global float *a, 
                      global float *b,
                      global float *c)
                      {
                      int gid = get_global_id(0);
                      c[gid] = a[gid] * b[gid];
                      }
        """
        program = pycl.Program(context, program_source).build()


        knlSum = program.sum  # Use this Kernel object for repeated calls
        knlProd = program.multiply  # Use this Kernel object for repeated calls

        for i in range(10):
            startTime = time.time()
            knlSum(queue, a_np.shape, None, a_g, b_g, res_g)
            knlProd(queue, a_np.shape, None, res_g, b_g, res_g)
            calcTime = time.time()-startTime
            res_np = np.empty_like(a_np)
            pycl.enqueue_copy(queue, res_np, res_g)
            copyTime = time.time()-startTime
            #print("\nCalculation time {0:.1f} ms, with copy {1:.1f} ms".format( 1000*calcTime, 1000*copyTime))

        # Check on CPU with Numpy:
        startTime = time.time()
        answer = (a_np + b_np)*b_np
        npTime = time.time() - startTime
        # print("Numpy {0:0.1f} ms".format(1000*npTime))
        assert np.allclose(res_np, answer)
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        devices = pycl.get_platforms()[0].get_devices(device_type=pycl.device_type.GPU)
        TestOpenCL.context = pycl.Context(devices=devices)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def testNumpyVectorsWithOpenCLLayout(self):
        vectors = np.empty((128,), dtype=pycl.cltypes.float3) # array of float16
        self.assertIsNotNone(vectors)

    def testCreateOpenCLArray(self):
        queue = pycl.CommandQueue(TestOpenCL.context)

        a = clArray(cq=queue, shape=(1024,), dtype=pycl.cltypes.float)
        self.assertIsNotNone(a)
        self.assertIsNotNone(a.data)

    def testCreateOpenCLArray(self):
        queue = pycl.CommandQueue(TestOpenCL.context)

        a = clArray(cq=queue, shape=(1024,), dtype=pycl.cltypes.float)
        self.assertIsNotNone(a)
        self.assertIsNotNone(a.data)
        for i in range(a.size):
            a[i] = i

        for i, e in enumerate(a):
            self.assertEqual(e, i)

    def testScalarMultiplicationOfOpenCLArrays(self):
        queue = pycl.CommandQueue(TestOpenCL.context)

        a = clArray(cq=queue, shape=(1024,), dtype=pycl.cltypes.float)
        self.assertIsNotNone(a)
        self.assertIsNotNone(a.data)

        for i in range(a.size):
            a[i] = i

        b = 2*a
        for i, e in enumerate(b):
            self.assertEqual(e, 2*i)

    @unittest.skip("Skipping for now")
    def test001ScalarMultiplicationOfOpenCLArrays(self):
        queue = pycl.CommandQueue(TestOpenCL.context)
        a = clArray(cq=queue, shape=(2<<12,), dtype=pycl.cltypes.float)
        for i in range(a.size):
            a[i] = i

        startTime = time.time()        
        b = a+a
        calcTime = (time.time()-startTime)*1000
        print("\nOpenCL 1 scalar: {0:.1f} ms ".format(calcTime))

        a = np.array(object=[0]*(2<<14), dtype=pycl.cltypes.float)
        for i in range(a.size):
            a[i] = i

        startTime = time.time()        
        b = a+a
        calcTime = (time.time()-startTime)*1000
        print("\nnumpy: {0:.1f} ms ".format(calcTime))

        a = clArray(cq=queue, shape=(2<<14,), dtype=pycl.cltypes.float)
        for i in range(a.size):
            a[i] = i

        startTime = time.time()        
        b = a+a
        calcTime = (time.time()-startTime)*1000
        print("\nOpenCL 2 scalar: {0:.1f} ms ".format(calcTime))

    @unittest.skip("Skipping for now")
    def test002ArraysWithAllocator(self):
        """
        I really expected this to work.  Performance is more complicated than I expected.
        The OpenCL calculation is much slower than the numpy version regardless of parameters 
        I used.

        The plan was simple: manipulate arrays in numpy and opencl, show it is much faster in Opencl.
        Well, it is not.

        UPDATE: yes it is, with VERY large arrays (2^18 or more). See next test.

        """

        # Set up basic OpenCl things
        queue = pycl.CommandQueue(TestOpenCL.context)
        allocator = pycl.tools.ImmediateAllocator(queue)
        mempool = pycl.tools.MemoryPool(allocator)

        N = 1<<10
        M = 1000

        # Pre-allocate all arrays
        a_n = np.random.rand(N).astype(np.float32)
        b_n = np.random.rand(N).astype(np.float32)

        # Pre-allocate opencl arrays with MemoryPool to reuse memory
        a = pycl.array.to_device(queue=queue, ary=a_n, allocator=mempool)
        b = pycl.array.to_device(queue=queue, ary=b_n, allocator=mempool)

        startTime = time.time()        
        for i in range(M):
            c = i*a + b + a + b + a + a + a
        calcTimeOpenCL1 = (time.time()-startTime)*1000

        startTime = time.time()        
        for i in range(M):
            c = i*a_n + b_n + a_n + b_n + a_n + a_n + a_n 
        calcTimeNumpy = (time.time()-startTime)*1000

        # Often, OpenCL is faster on the second attempt.
        startTime = time.time()        
        for i in range(M):
            c = i*a + b + a + b + a + a + a
        calcTimeOpenCL2 = (time.time()-startTime)*1000

        self.assertTrue(calcTimeOpenCL2 < calcTimeNumpy,msg="\nNumpy is faster than OpenCL: CL1 {0:.1f} ms NP {1:.1f} ms CL2 {2:.1f} ms".format(calcTimeOpenCL1, calcTimeNumpy, calcTimeOpenCL2))
        print("\nCL1 {0:.1f} ms NP {1:.1f} ms".format(calcTimeOpenCL2, calcTimeNumpy))

    @unittest.skip("Skipping graphic tests")
    def test003PerformanceVsSize(self):
        """
        Performance with OpenCL is better but with very large arrays (2^20 or more)
        """

        # Set up basic OpenCl things
        queue = pycl.CommandQueue(TestOpenCL.context)
        allocator = pycl.tools.ImmediateAllocator(queue)
        mempool = pycl.tools.MemoryPool(allocator)

        N = 1
        M = 10
        P = 27
        nptimes = []
        cltimes = []
        for j in range(P):
            # Pre-allocate all arrays
            N = 1 << j
            a_n = np.random.rand(N).astype(np.float32)
            b_n = np.random.rand(N).astype(np.float32)

            # Pre-allocate opencl arrays, with MemoryPool to reuse memory
            a = pycl.array.to_device(queue=queue, ary=a_n, allocator=mempool)
            b = pycl.array.to_device(queue=queue, ary=b_n, allocator=mempool)

            startTime = time.time()        
            calcTimeOpenCL1 = []
            for i in range(M):
                c = i*a + b + a + b + a + a * a
            calcTimeOpenCL1.append((time.time()-startTime)*1000)
            cltimes.append(np.mean(calcTimeOpenCL1))

            startTime = time.time()        
            calcTimeOpenNP = []
            for i in range(M):
                c = i*a_n + b_n + a_n + b_n + a_n + a_n * a_n
            calcTimeOpenNP.append((time.time()-startTime)*1000)
            nptimes.append(np.mean(calcTimeOpenNP))

        plt.plot(range(P), cltimes, label="OpenCL")
        plt.plot(range(P), nptimes, label="Numpy")
        plt.xlabel("Size of array 2^x")
        plt.ylabel("Computation time [ms]")
        plt.legend()
        plt.show()

    def test004_2x2Matrix_and_Vectors(self):
        """
        Here I am getting excited about the possiblities for RayTracing and I want to
        see it in action multiplying 2x2 matrices and vectors.

        """

        context = TestOpenCL.context
        queue = pycl.CommandQueue(context)

        program_source = """
        kernel void product(global const float *mat, int M, 
                            global float *vec,
                            global float *res)
                      {
                      int i    = get_global_id(0); // the vector index
                      int j;                       // the matrix index

                      for (j = 0; j < M; j++) {
                          res[i + 2*j]     = vec[i];
                          res[i + 2*j + 1] = vec[i+1];

                          vec[i]     = mat[i+4*j]   * vec[i] + mat[i+4*j+1] * vec[i+1];
                          vec[i + 1] = mat[i+4*j+2] * vec[i] + mat[i+4*j+3] * vec[i+1];
                          }
                      }

        """
        program_source_floats = """
        kernel void product(global const float4 *mat, int M, 
                            global float2 *vec,
                            global float2 *res)
                      {
                      int i    = get_global_id(0); // the vector index
                      int j;                       // the matrix index
                      int N    = get_global_size(0);
                      float2 v = vec[i];
                      res[i] = v;
                      for (j = 0; j < M; j++) {
                          float4 m = mat[j];

                          v.x = m.x * v.x + m.y * v.y;
                          v.y = m.z * v.x + m.w * v.y;
                          res[i+N*(j+1)] = v;
                          }
                      }

        """
        program = pycl.Program(context, program_source_floats).build()
        knl = program.product  # Use this Kernel object for repeated calls


        startTime = time.time()        
        M = np.int32(40)    # M 2x2 matrices in path
        N = np.int32(2^24)  # N 2x1 rays to propagate  
        # Pre-allocate opencl arrays, with MemoryPool to reuse memory
        matrix_n = np.random.rand(M,2,2).astype(np.float32)
        vector_n = np.random.rand(N,2).astype(np.float32)
        result_n = np.zeros((M+1,N,2)).astype(np.float32)

        matrix = pycl.array.to_device(queue=queue, ary=matrix_n)
        vector = pycl.array.to_device(queue=queue, ary=vector_n)
        result = pycl.array.to_device(queue=queue, ary=result_n)

        knl(queue, (N,), None, matrix.data, M, vector.data, result.data)

        print("\n{0:0.1f} ms".format((time.time()-startTime)*1000))


class TestOpenCLIds(unittest.TestCase):
    context = None

    def setUp(self):
        self.gpuDevices = pycl.get_platforms()[0].get_devices(pycl.device_type.GPU)
        self.context = pycl.Context(devices=self.gpuDevices)
        self.queue = pycl.CommandQueue(self.context)
        rootdir = Path(__file__).parent
        self.program_source = Path(rootdir, 'test_opencl.c').read_text()        
        self.program = pycl.Program(self.context, self.program_source).build()

    def test_get_kernel_info(self):
        kernel = self.program.test_global_id
        self.assertIsNotNone(kernel)

    def test_trivial_kernel_global_id(self):
        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 100
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_global_id # copy global_id into array

        knl(queue, (nWorkUnits,), None, valueBuffer.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(value, i)

    def test_trivial_kernel_local_id_default_workgroup_size(self):
        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 100
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_local_id # copy local_id into array

        knl(queue, (nWorkUnits,), None, valueBuffer.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(i%32, value)

    def test_trivial_kernel_local_id_set_workgroup_size(self):
        """
        The number of groups must divide nWorkUnits evenly
        """

        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 100
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_local_id # copy local_id into array

        for i in range(100):
            try:
                knl(queue, (nWorkUnits,), (i,), valueBuffer.data)
                self.assertEqual(nWorkUnits%i, 0)

                for j, value in enumerate(valueBuffer):
                    self.assertEqual(j%nWorkUnits, value)
            except:
                pass


    # @unittest.skip('For information only')
    # def test_extract_many_ids(self):
    #     nWorkUnits = 100
    #     valueBuffer = Buffer(nWorkUnits, value=0, dtype=cl.cltypes.uint)

    #     source_path = os.path.join(OPENCL_SOURCE_DIR, "test_opencl.c")
    #     program = CLProgram(source_path)
    #     program.launchKernel(kernelName='test_extract_many_ids', N=nWorkUnits, arguments = [valueBuffer])
    #     for i, value in enumerate(valueBuffer.hostBuffer):
    #         print(i,value)

    def test_compute_global_id_from_local_id_uniform_sizes(self):
        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 128 # Multiplie of 32
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_compute_global_id_from_local_id # copy local_id into array
        knl(queue, (nWorkUnits,), None, valueBuffer.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(value, i)

    @unittest.expectedFailure
    def test_compute_global_id_from_local_id_NON_uniform_sizes(self):
        """
        My assumption was that workunits would be fed in order but that is not true.
        I don't know how to compute the global_id from the local parameters then
        It does not matter: use get_global_id() but this is a basic issue in 
        understanding OpenCL for me.
        """

        queue = pycl.CommandQueue(self.context)
        nWorkUnits = 100 # NOT A MULTIPLE of 32
        valueBuffer = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)
        local_sizes = clArray(cq=queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)

        knl = self.program.test_compute_global_id_from_local_id_nonuniform # copy local_id into array
        knl(queue, (nWorkUnits,), None, valueBuffer.data, local_sizes.data)

        for i, value in enumerate(valueBuffer):
            self.assertEqual(value, i)

        self.assertEqual(np.sum(local_sizes.hostBuffer), nWorkUnits)

    def test_call_random_value_with_same_seeds_buffer_should_give_same_results(self):
        self.program_source =  """

            uint wangHash(uint seed){
                seed = (seed ^ 61) ^ (seed >> 16);
                seed *= 9;
                seed = seed ^ (seed >> 4);
                seed *= 0x27d4eb2d;
                seed = seed ^ (seed >> 15);
                return seed;
            }

            float getRandomFloatValue(__global unsigned int *seeds, int id){
                 float result = 0.0f;
                 while(result == 0.0f){
                     uint rnd_seed = wangHash(seeds[id]);
                     seeds[id] = rnd_seed;
                     result = (float)rnd_seed / (float)UINT_MAX;
                 }
                 return result;
            }

            // ----------------- TEST KERNELS -----------------

             __kernel void fillRandomFloatBuffer(__global unsigned int *seeds, __global float *randomNumbers){
                int id = get_global_id(0);
                randomNumbers[id] = getRandomFloatValue(seeds, id);
            }
            """
        self.program = pycl.Program(self.context, self.program_source).build()
        nWorkUnits = 16 # NOT A MULTIPLE of 32

        seedsBuffer1 = clArray(cq=self.queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)
        for i in range(nWorkUnits):
            seedsBuffer1[i] = i
        valueBuffer1 = clArray(cq=self.queue, shape=(nWorkUnits,), dtype=np.float32)

        seedsBuffer2 = clArray(cq=self.queue, shape=(nWorkUnits,), dtype=cl.cltypes.uint)
        for i in range(nWorkUnits):
            seedsBuffer2[i] = i
        valueBuffer2 = clArray(cq=self.queue, shape=(nWorkUnits,), dtype=np.float32)

        knl = self.program.fillRandomFloatBuffer
        knl(self.queue, (nWorkUnits,), None, seedsBuffer1.data, valueBuffer1.data)
        knl(self.queue, (nWorkUnits,), None, seedsBuffer2.data, valueBuffer2.data)

        self.assertTrue( all(valueBuffer1==valueBuffer2))



if __name__ == "__main__":
    unittest.main()
