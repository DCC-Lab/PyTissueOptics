import unittest

from pytissueoptics.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.logger = Logger()

    def testWhenLogEnergyPoint_shouldStorePoint(self):
        position = (0, 0)
        energy = 1
        
        self.logger.logEnergy([position], [energy])
        
        self.assertEqual(self.logger.getEnergy(), ([position], [energy]))

    def testWhenLogMultipleEnergyPoints_shouldStoreAllPoints(self):
        positions = [(0, 0), (1, 1)]
        energy = [0.5, 1]
        
        self.logger.logEnergy(positions, energy)
        
        self.assertEqual(self.logger.getEnergy(), (positions, energy))

    def testGivenALoggerWithEnergyPoints_whenLogEnergy_shouldAppendToPreviousPoints(self):
        positions = [(0, 0), (1, 1)]
        energy = [0.5, 1]
        self.logger.logEnergy(positions, energy)
        newPositions = [(2, 2)]
        newEnergy = [2]
        
        self.logger.logEnergy(newPositions, newEnergy)

        self.assertEqual(self.logger.getEnergy(), (positions + newPositions, energy + newEnergy))


if __name__ == '__main__':
    unittest.main()
