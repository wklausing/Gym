'''
In this environment the CP-Agent has to buy contingent from edge-servers and steer clients to edge-servers. 
Rewards are defined by QoE metrics from client which are done by Sabre.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import math
from sabreEnv.sabre.sabre import Sabre
from collections import namedtuple

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

        self.buyContingent = [100] * edgeServers
        self.manifest = clients * edgeServers * [edgeServers]
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.manifest)

    def _get_obs(self):
        return {"clientsLocations": self.clientsLocations, "edgeServerLocations": self.edgeServerLocations
                , "edgeServerPrices": self.edgeServerPrices, "time": self.time
                , "money": self.money}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset env variables
        self.time = np.array([0], dtype='int')

        # Reset CP-Agent
        self.money = np.array([100_000], dtype='int')

        # Reset edge servers
        self.edgeServerLocations = self.np_random.integers(0, self.gridSize, size=self.edgeServerCount, dtype=int)
        self.edgeServerPrices = np.round(np.random.uniform(0, 10, size=self.edgeServerCount), 2)
        self.edgeServers = []
        for e in range(self.edgeServerCount):
            self.edgeServers.append(EdgeServer(self.edgeServerLocations[e], self.edgeServerPrices[e]))

        # Reset clients
        self.clientsLocations = self.np_random.integers(0, self.gridSize, size=self.clientCount, dtype=int)
        self.clients = []
        for c in range(self.clientCount):
            self.clients.append(Client(self.clientsLocations[c], self.edgeServers))

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        time = self.time.item()
        money = self.money.item()
        reward = 0

        # An episode is done when the CP is out of money or time is over
        terminated = money <= 0 or time >= 7_200

        if not terminated:
            # Buy contigent
            buyContigent = action[:len(self.buyContingent)] 
            for index, edgeServer in enumerate(self.edgeServers):
                money = edgeServer.sellContigent(money, buyContigent[index])
            
            # Create manifest
            manifest = action[len(self.buyContingent):] 
            for _, client in enumerate(self.clients):
                if client.alive:
                    client.manifest = manifest[:4]
                    manifest = manifest[4:]
                    money += client.fetchContent(time)
                    reward += client.getQoE()

            # Remove dead clients
            for i, client in enumerate(self.clients):
                if client.alive == False:
                    del self.clients[i]

        observation = self._get_obs()
        info = self._get_info()

        time += 1

        self.time = np.array([time], dtype='int')
        self.money = np.array([money], dtype='int')

        return observation, reward, terminated, False, info
    
    ### UTILS ###
    def distance(self, position1, position2):
        '''
        Calculates distance between two points. Used for latency calculation.
        '''
        position1 = position1 % 100, position1 // 100
        position2 = position2 % 100, position2 // 100
        distance = math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2) # Euklidean distance
        return round(distance, 2)


class Client():

    manifest = []

    def __init__(self, location, manifest=[]):
        self.alive = True
        self.location = location
        self.manifest = manifest
        self.idxCDN = 0
        self.edgeServer = manifest[self.idxCDN]
        self.edgeServer.addClient(self)
        self.stillStreaming = True
        self.latency = 100
        self.time = 0

        # Sabre implementation
        self.sabre = Sabre(verbose=False)        

    def fetchContent(self, time):
        '''
        If fetch origin can delivier than return 1. If not than increase idxCDN to select next server and return -1.
        Here Sabre should be used to get a real reward based on QoE. 
        '''
        # Distance between client and edge-server. Used for latency calculation. TODO let distance determine latency
        latency = self.latency

        # Bandwidth
        bandwidth = self.bandwidth

        # Add network trace to client
        self.time = self.time - time
        self.sabre.network._add_network_condition(duration_ms=time, bandwidth_kbps=bandwidth, latency_ms=latency)

        # For finance stuff of CDNs
        if self.edgeServer.soldContigent > 0:
            self.edgeServer.soldContigent -= 1
            return 1
        else:
            self.idxCDN += 1 
            self.idxCDN = self.idxCDN % len(self.manifest)
            return -1
        
    def getQoE(self):
        '''
        Here QoE will be computed with Sabre.
        '''
        sabreResult = self.sabre.downloadSegment()
        if sabreResult['done'] == False and len(sabreResult) == 5:
            result = sabreResult
            qoe = result['time_average_played_bitrate'] - result['time_average_bitrate_change'] - result['time_average_rebuffer_events']
            return qoe
        elif sabreResult['done']:
            self.alive = False
            return 0
        else:
            return 0
    
    def setBandwidth(self, bandwidth):
        self.bandwidth = bandwidth


class EdgeServer:

    def __init__(self, location, price=1, bandwidth_kbps=1000):
        self.location = location
        self.price = price
        self.soldContigent = 0
        self.bandwidth_kbps = bandwidth_kbps
        self.clients = []

    def sellContigent(self, cpMoney, amount):
        leftOverMoney = 0
        if cpMoney >= amount * self.price:
            cpMoney -= amount * self.price
            self.soldContigent += amount
            leftOverMoney = amount * self.price
        else:
            leftOverMoney = cpMoney
        return round(leftOverMoney, 2)
    
    def addClient(self, client):
        self.clients.append(client)
        bandwidth = self.bandwidth_kbps / len(self.clients)
        for c in self.clients:
            c.setBandwidth(bandwidth)

    def removeClient(self, client):
        self.clients.remove(client)
        bandwidth = self.bandwidth_kbps / len(self.clients)
        for c in self.clients:
            c.setBandwidth(bandwidth)
    
    @property
    def bandwidth(self):
        return self.bandwidth_kbps


if __name__ == "__main__":
    print('### Start ###')
    env = GymSabreEnv(render_mode="human")
    observation, info = env.reset()

    for i in range(2_000):
        print(i)
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print('### Done ###')
