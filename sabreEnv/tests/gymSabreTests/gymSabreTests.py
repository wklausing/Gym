import unittest

from sabreEnv import GymSabreEnv
from gymnasium.utils.env_checker import check_env

class TestMainFunction(unittest.TestCase):


    def testCheck_env(self):
        try:
            check_env(env=GymSabreEnv(render_mode="human", maxActiveClients=1, totalClients=1, cdnLocations=1), warn=True, skip_render_check=True)
        except:
            self.fail("check_env() raised Exception!")

    def testBandwidth(self):
        '''
        Testing bandwidth behaviour of an edge server.
        For this x clients are connected to the edge server.
        Bandwidth should be split between the two clients.
        Expects only 1 edgeServer!
        '''
        gymSabre = GymSabreEnv(clients=20, cdnLocations=1)
        gymSabre.reset()

        totalCdnBandwidth = 0
        for cdn in gymSabre.cdns:
            totalCdnBandwidth += cdn.bandwidth

        totalClientBandwidth = 0
        for client in gymSabre.clients:
            totalClientBandwidth += client.bandwidth

        self.assertEqual(totalCdnBandwidth, totalClientBandwidth)

    def testStaticManifest(self):
        '''
        Testing the static manifest switching. 
        '''
        gymSabre = GymSabreEnv(clients=50, cdnLocations=2)
        gymSabre.reset()
        gymSabre.cdns[0].soldContigent = 100
        gymSabre.cdns[1].soldContigent = 100

        env = GymSabreEnv()
        observation, info = env.reset()

        for _ in range(1000):
            action = env.action_space.sample()  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated or gymSabre.cdns[0].soldContigent == 0 or gymSabre.cdns[1].soldContigent == 0:
                break
        env.close()
        self.assertEqual(gymSabre.cdns[0].soldContigent, gymSabre.cdns[1].soldContigent)
                         

if __name__ == '__main__':
    unittest.main()
