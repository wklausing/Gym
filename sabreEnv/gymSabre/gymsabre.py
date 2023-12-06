'''
In this environment the CP-Agent has to buy contingent from edge-servers and steer clients to edge-servers. 
Rewards are defined by QoE metrics from client which are done by Sabre.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from sabreEnv.sabre.sabre import Sabre

from gymnasium.wrappers import RecordEpisodeStatistics

gym.logger.set_level(20) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridSize = 100*100, edgeServers = 4, clients = 10):

        # Env variables
        self.gridSize = gridSize

        # CP-Agent variables
        
        # CDN variables
        self.edgeServerCount = edgeServers
        self.edgeServerLocations = np.ones(edgeServers, dtype=int)
        self.edgeServerPrices = np.ones(edgeServers, dtype=int)

        # Client variables
        self.clientCount = clients
        self.clientsLocations = np.ones(clients, dtype=int)

        # Observation space for CP agent. Contains location of clients, location of edge-servers, pricing of edge-server, and time in seconds.        
        self.observation_space = spaces.Dict(
            {
                'clientsLocations': gym.spaces.MultiDiscrete([gridSize] * clients),
                'edgeServerLocations': gym.spaces.MultiDiscrete([gridSize] * edgeServers),
                'edgeServerPrices': spaces.Box(0, 10, shape=(edgeServers,), dtype=float),
                'time': spaces.Box(0, 100_000, shape=(1,), dtype=int),
                'money': spaces.Box(0, 100_000, shape=(1,), dtype=int),
            }
        )
 
        # Action space for CP agent. Contains buy contigent and manifest for clients.
        self.buyContingent = [100] * edgeServers
        self.manifest = clients * edgeServers * [edgeServers]
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.manifest)

    def _get_obs(self):
        return {'clientsLocations': self.clientsLocations, 'edgeServerLocations': self.edgeServerLocations
                , 'edgeServerPrices': self.edgeServerPrices, 'time': self.time
                , 'money': self.money}

    def _get_info(self, reward):
        self.sumReward += reward
        return {'money': self.money, 'time': self.time, 'sumReward': self.sumReward}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset env variables
        self.sumReward = 0
        self.time = np.array([0], dtype='int')

        # Reset CP-Agent
        self.money = np.array([100_000], dtype='int')

        # Reset edge servers
        self.edgeServerLocations = self.np_random.integers(0, self.gridSize, size=self.edgeServerCount, dtype=int)
        self.edgeServerPrices = np.round(np.random.uniform(0, 10, size=self.edgeServerCount), 2)
        self.edgeServers = []
        for e in range(self.edgeServerCount):
            self.edgeServers.append(EdgeServer(e, self.edgeServerLocations[e], self.edgeServerPrices[e]))

        # Reset clients
        self.clientsLocations = self.np_random.integers(0, self.gridSize, size=self.clientCount, dtype=int)
        self.clients = []
        for c in range(self.clientCount):
            self.clients.append(Client(c ,self.clientsLocations[c], self.edgeServers))

        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        time = self.time.item()
        money = self.money.item()
        reward = 0

        # An episode is done when the CP is out of money or time is over
        terminated = money <= 0 or time >= 7_200 or len(self.clients) == 0

        if not terminated:
            # Buy contigent
            buyContigent = action[:len(self.buyContingent)] 
            for i, edgeServer in enumerate(self.edgeServers):
                money = edgeServer.sellContigent(money, buyContigent[i])
            
            # Create manifest
            manifest = action[len(self.buyContingent):] 
            for i, client in enumerate(self.clients):
                if client.alive:
                    if not hasattr(client, 'manifest'): client.setManifest(manifest[:4])
                    result = client.fetchContent(time)
                    if result != {}:
                        reward += result['qoe']
                        if not client.alive:
                            gym.logger.info('Client %s died' % client.id)
                    else:
                        gym.logger.info('Client %s could not fetch content' % client.id)
                else:
                    del self.clients[i]                    

        observation = self._get_obs()

        time += 1

        self.time = np.array([time], dtype='int')
        self.money = np.array([money], dtype='int')

        info = self._get_info(reward)
        return observation, reward, terminated, False, info


class Client():
    '''
    Manifest is list with the ids of edge-server.
    cdns is list of edge-servers.
    '''

    def __init__(self, id, location, cdns):
        self.id = id
        self.alive = True
        self.location = location
        self.time = 0
        self.cdns = cdns
        self.bandwidth = 0

        # Sabre implementation
        self.sabre = Sabre(verbose=False)   

    def setManifest(self, manifest):
        self.manifest = manifest
        self.idxCDN = 0
        self.edgeServer = self.cdns[self.idxCDN]
        self.edgeServer.addClient(self)

    def fetchContent(self, time):
        '''
        If fetch origin can delivier than return 1. If not than increase idxCDN to select next server and return -1.
        Here Sabre should be used to get a real reward based on QoE. 

        The CP gets money from clients whenever they fetch content.
        '''
        # Distance between client and edge-server. Used for latency calculation.
        latency = self.determineLatency(self.location, self.cdns[self.idxCDN].location)

        # Bandwidth
        bandwidth = self.edgeServer.currentBandwidth

        # Each time a client fetches content it pays the edge-server.
        if self.edgeServer.contigent > 0:
            # Add network trace to client
            self.time = self.time - time
            self.sabre.network._add_network_condition(duration_ms=time, bandwidth_kbps=bandwidth, latency_ms=latency)
        else:
            gym.logger.info('Not enough contingent for client %s from CDN at location %s' % (self.id, self.edgeServer.location))
            self._changeCDN()
            self.sabre.network._remove_network_condition()

        qoe = self.getQoE()
        if qoe != {}:
            size = qoe['size']
            self.edgeServer.contigent -= size / 1000

        return self.getQoE()
        
    def getQoE(self):
        '''
        Here QoE will be computed with Sabre.
        '''
        result = self.sabre.downloadSegment()
        if result['done'] == False and len(result) == 6:
            qoe = result['time_average_played_bitrate'] - result['time_average_bitrate_change'] - result['time_average_rebuffer_events']
            return {'qoe': qoe, 'size': result['size']}
        elif result['done']:
            self.setDone()
            return {}
        else:
            return {}
        
    def _changeCDN(self):
        '''
        Change CDN if i.e. QoE is bad.
        Currently it start from the beginning of the manifest when it reached the end.
        '''
        self.idxCDN += 1
        if self.idxCDN == len(self.manifest):
            self.idxCDN -= 1
        else:
            self.edgeServer.removeClient(self)
            self.edgeServer = self.cdns[self.manifest[self.idxCDN]]
            self.edgeServer.addClient(self)
            gym.logger.info('Client %s changed to CDN %s' % (self.id, self.idxCDN))

    def setDone(self):
        self.alive = False
        self.edgeServer.removeClient(self)

    def determineLatency(self, position1, position2):
        '''
        Calculates distance between two points. Used to calcualte latency.
        '''
        position1 = position1 % 100, position1 // 100
        position2 = position2 % 100, position2 // 100
        distance = round(math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2), 2) # Euklidean distance
        distance = round(distance * 5, 2)
        return distance


class EdgeServer:

    def __init__(self, id, location, price=1, bandwidth_kbps=1000):
        self.id = id
        self.location = location
        self.price = price
        self.contigent = 0
        self.bandwidth_kbps = bandwidth_kbps
        self.clients = []

    def sellContigent(self, cpMoney, amount):
        leftOverMoney = 0
        if cpMoney >= amount * self.price:
            cpMoney -= amount * self.price
            self.contigent += amount
            leftOverMoney = amount * self.price
        else:
            leftOverMoney = cpMoney
        return round(leftOverMoney, 2)
    
    def addClient(self, client):
        self.clients.append(client)
        self.currentBandwidth = self.bandwidth_kbps / len(self.clients)

    def removeClient(self, client):
        self.clients.remove(client)
        if len(self.clients) == 0: return
        self.currentBandwidth = self.bandwidth_kbps / len(self.clients)
    
    @property
    def bandwidth(self):
        return self.bandwidth_kbps


if __name__ == "__main__":
    print('### Start ###')
    env = GymSabreEnv(render_mode="human", clients=2)
    env = RecordEpisodeStatistics(env)
    observation, info = env.reset()

    for i in range(7_200):
        action = env.action_space.sample() # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        if reward != 0: print(reward)

    env.close()
    print('### Done ###')
