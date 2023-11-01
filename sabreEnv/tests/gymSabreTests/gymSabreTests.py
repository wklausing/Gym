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

    def testStaticManifest(self):
        '''
        Testing the static manifest switching. 
        '''
        gymSabre = GymSabreEnv(clients=50, edgeServers=2)
        gymSabre.reset()
        gymSabre.edgeServers[0].soldContigent = 100
        gymSabre.edgeServers[1].soldContigent = 100

        env = GymSabreEnv()
        observation, info = env.reset()

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated or gymSabre.edgeServers[0].soldContigent == 0 or gymSabre.edgeServers[1].soldContigent == 0:
                break
        env.close()
        self.assertEqual(gymSabre.edgeServers[0].soldContigent, gymSabre.edgeServers[1].soldContigent)
                         

if __name__ == '__main__':
    unittest.main()
