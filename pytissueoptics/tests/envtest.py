# append module root directory to sys.path
import sys
import os
sys.path.insert(0,
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__)
                        )
                    )
                )
                )

import io
from contextlib import redirect_stdout
import unittest
import tempfile
import matplotlib.pyplot as plt
import random
from pytissueoptics.vector import Vector


class PyTissueTestCase(unittest.TestCase):
    tempDir = os.path.join(tempfile.gettempdir(), "tempDir")

    def __init__(self, tests=()):
        super(PyTissueTestCase, self).__init__(tests)

    def tearDown(self) -> None:
        self.clearMatplotlibPlots()

    @classmethod
    def clearMatplotlibPlots(cls):
        plt.clf()
        plt.cla()
        plt.close()

    def assertDoesNotRaise(self, func, exceptionType=None, *funcArgs, **funcKwargs):
        returnValue = None
        if exceptionType is None:
            exceptionType = Exception
        try:
            returnValue = func(*funcArgs, **funcKwargs)
        except exceptionType as e:
            self.fail(f"An exception was raised:\n{e}")
        # Don't handle exceptions not in exceptionType
        return returnValue

    def assertPrints(self, func, out, stripOutput: bool = True, *funcArgs, **funcKwargs):
        @redirectStdOutToFile
        def getOutput():
            func(*funcArgs, **funcKwargs)

        value = getOutput()
        if stripOutput:
            value = value.strip()
        self.assertEqual(value, out)

    def randomVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2-1
        return Vector(x,y,z)

    def randomUnitVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2-1
        return Vector(x,y,z).normalized()

    def randomPositiveZVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = random.random()*2
        return Vector(x,y,z)

    def randomNegativeZVector(self):
        x = random.random()*2-1
        y = random.random()*2-1
        z = -random.random()*2
        return Vector(x,y,z)

    def randomVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( Vector(x,y,z) )

    def randomUnitVectors(self, N):
        vectors = []
        for i in range(N):
            x = random.random()*2-1
            y = random.random()*2-1
            z = random.random()*2-1
            vectors.append( UnitVector(x,y,z) )



    @classmethod
    def createTempDirectory(cls):
        if os.path.exists(cls.tempDir):
            cls.deleteTempDirectory()
        os.mkdir(cls.tempDir)

    @classmethod
    def deleteTempDirectory(cls):
        if os.path.exists(cls.tempDir):
            for file in os.listdir(cls.tempDir):
                os.remove(os.path.join(cls.tempDir, file))
            os.rmdir(cls.tempDir)

    @classmethod
    def setUpClass(cls) -> None:
        cls.createTempDirectory()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.deleteTempDirectory()

    def tempFilePath(self, filename="temp.dat") -> str:
        return os.path.join(RaytracingTestCase.tempDir, filename)


def redirectStdOutToFile(_func=None, file=None, returnOnlyValue: bool = True):
    if file is None:
        file = io.StringIO()

    def redirectStdOut(func):
        def wrapperRedirectStdOut(*args, **kwargs):
            with redirect_stdout(file):
                func(*args, **kwargs)

            return file.getvalue() if returnOnlyValue else file

        return wrapperRedirectStdOut

    if _func is None:
        return redirectStdOut
    return redirectStdOut(_func)


def main():
    unittest.main()


def skip(reason: str):
    return unittest.skip(reason)


def skipIf(condition: object, reason: str):
    return unittest.skipIf(condition, reason)


def skipUnless(condition: object, reason: str):
    return unittest.skipUnless(condition, reason)


def expectedFailure(func):
    return unittest.expectedFailure(func)


