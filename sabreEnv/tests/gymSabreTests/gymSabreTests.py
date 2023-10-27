import unittest
import sys

from sabreEnv import GymSabreEnv, SabreActionWrapper

class TestMainFunction(unittest.TestCase):

    def testBandwidth(self):
        '''
        Testing bandwidth behaviour of an edge server.
        For this x clients are connected to the edge server.
        Bandwidth should be split between the two clients.
        Expects only 1 edgeServer!
        '''
        gymSabre = GymSabreEnv(clients=20, edgeServers=1)
        gymSabre.reset()

        totalCdnBandwidth = 0
        for cdn in gymSabre.edgeServers:
            totalCdnBandwidth += cdn.bandwidth

        totalClientBandwidth = 0
        for client in gymSabre.clients:
            totalClientBandwidth += client.bandwidth

        self.assertEqual(totalCdnBandwidth, totalClientBandwidth)
                         

if __name__ == '__main__':
    unittest.main()
