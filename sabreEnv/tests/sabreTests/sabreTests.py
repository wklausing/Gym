import unittest

from sabreEnv import Sabre

class TestMainFunction(unittest.TestCase):

    def testBandwidth(self):
        '''
        Testing Sabre
        '''
        sabre = Sabre()
        result = sabre.testing()
        print(result)

        self.assertEqual(1, 1)
                         

if __name__ == '__main__':
    unittest.main()
