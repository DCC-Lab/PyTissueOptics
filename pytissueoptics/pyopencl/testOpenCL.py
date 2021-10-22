import pyopencl as pycl
import unittest


class TestOpenCL(unittest.TestCase):
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

    def testProgramSource(self):
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


if __name__ == "__main__":
    unittest.main()
