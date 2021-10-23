import pyopencl as pycl
import unittest


class TestOpenCL(unittest.TestCase):
	def test01Import(self):
		self.assertIsNotNone(pycl)

	def test02AtLeast1(self):
		self.assertTrue(len(pycl.get_platforms()) > 0)

	def test03AtLeast1Device(self):
		platform = pycl.get_platforms()[0]
		devices = platform.get_devices()
		self.assertTrue(len(devices) > 0)

	def test04Context(self):
		platform = pycl.get_platforms()[0]
		devices = platform.get_devices()
		context = pycl.Context(devices=devices)
		self.assertIsNotNone(context)

if __name__ == "__main__":
	unittest.main()
