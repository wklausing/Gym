'''
In this environment the CP-Agent goals is the provide all clients with a good QoE. For doing so it needs to buy contigents 
from CDNs and create manifests for clients. The reward is the cumulative reward of all clients and the money left at the CP-Agent.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gymnasium.wrappers import RecordEpisodeStatistics

from datetime import datetime

from sabreEnv.gymSabre.client import Client
from sabreEnv.gymSabre.cdn import EdgeServer
from sabreEnv.gymSabre.util import Util

import random

gym.logger.set_level(10) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridSize=100*100, serviceLocations=4, clients=10, saveData=False, contentSteering=False, ttl=500):
        # Util
        self.util = Util()
        
        # For recordings
        self.saveData = saveData
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cdnFilename = 'data/cdn_' + time + '.csv'
        self.cpFilename = 'data/cp_' + time + '.csv'

        # Env variables
        self.gridSize = gridSize

        # CP-Agent variables
                
        # CDN variables
        self.serviceLocationCount = serviceLocations
        self.serviceLocations = np.ones(serviceLocations, dtype=int)
        self.cdnPrices = np.ones(serviceLocations, dtype=int)

        # Client variables
        self.clientCount = clients
        self.contentSteering = contentSteering
        self.ttl = ttl

        # Observation space for CP agent. Contains location of clients, location of edge-servers, pricing of edge-server, and time in seconds.        
        self.observation_space = spaces.Dict(
            {
                'client': gym.spaces.Discrete(gridSize+1, start=-1),
                'clientsLocations': gym.spaces.MultiDiscrete([gridSize] * clients),
                'serviceLocations': gym.spaces.MultiDiscrete([gridSize] * serviceLocations),
                'cdnPrices': spaces.Box(0, 10, shape=(serviceLocations,), dtype=float),
                'time': spaces.Box(0, 100_000, shape=(1,), dtype=int),
                'money': spaces.Box(0, 100_000, shape=(1,), dtype=int),
            }
        )
 
        # Action space for CP agent. Contains buy contigent and manifest for clients.
        self.buyContingent = [100] * serviceLocations
        self.manifest = serviceLocations * [serviceLocations]
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.manifest)

    def _get_obs(self):
        # Client who receices a manifest
        clientManifest = -1
        for client in self.clients:
            if client.alive and client.needsManifest:
                clientManifest = client.location

        # clientsLocations
        clientsLocations = []
        self.clientCount
        for client in self.clients:
            clientsLocations.append(client.location)

        # Fill clientsLocations with -1 so that observation space doesn't change
        while len(clientsLocations) < self.clientCount:
            clientsLocations.append(-1)

        return {
            'client': np.array(clientManifest), 
            'clientsLocations': clientsLocations, 'serviceLocations': self.serviceLocations
                , 'cdnPrices': self.cdnPrices, 'time': self.time
                , 'money': self.money}

    def _get_info(self, reward):
        self.sumReward += reward
        return {'money': self.money, 'time': self.time, 'sumReward': self.sumReward}

    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # For recordings
        self.filename = 'data/gymSabre_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json'
        self.data = []
        self.util.episodeCounter += 1
        self.stepCounter = 0
        
        # Reset env variables
        self.sumReward = 0
        self.time = np.array([0], dtype='int')

        # Reset CP-Agent
        self.money = np.array([100], dtype='int')

        # Reset CDN servers
        self.serviceLocations = self.np_random.integers(0, self.gridSize, size=self.serviceLocationCount, dtype=int)
        self.cdnPrices = np.round(np.random.uniform(0.5, 2, size=self.serviceLocationCount), 2)
        self.edgeServers = []
        for e in range(self.serviceLocationCount):
            self.edgeServers.append(EdgeServer(self.util, e, self.serviceLocations[e].item(), self.cdnFilename, self.cdnPrices[e]))

        # Reset clients
        self.clients = []
        for c in range(self.clientCount):
            self.clients.append(Client(c , random.randint(0, self.gridSize), self.edgeServers, util=self.util, contentSteering=self.contentSteering, ttl=self.ttl))

        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        '''
        Here the step from CP agent is done, but also from CDN and clients.
        '''
        self.stepCounter += 1

        time = self.time.item()
        money = self.money.item()
        reward = 0

        # An episode is done when the CP is out of money or time is over
        allClientsDone = all(not client.alive for client in self.clients)
        if allClientsDone:
            print('All clients done.')
        elif money <= -1_000_000:
            print('Money is below -1_000_000.')
        elif time >= 7_200:
            print('Time is over.')
        terminated = money <= -1_000_000 or self.stepCounter >= 7_200 or allClientsDone

        if terminated:
            pass

        for client in self.clients:
           if not client.alive:
               pass

        # Saving data
        clients_to_remove = []
        for client in self.clients:
            if not client.alive or terminated:
                client.saveData(finalStep=self.saveData)
                clients_to_remove.append(client)
        for edgeServer in self.edgeServers:
            edgeServer.saveData(time, finalStep=terminated)

        # Remove clients
        for client in clients_to_remove:
            self.clients.remove(client)

        if terminated:
            print('Termination time:', time)
            pass
        else:
            # Buy contigent from CDNs
            buyContigent = action[:len(self.buyContingent)] 
            for i, edgeServer in enumerate(self.edgeServers):
                if hasattr(buyContigent[i], 'item'): 
                    money = edgeServer.sellContigent(money, buyContigent[i].item())
                else:
                    money = edgeServer.sellContigent(money, buyContigent[i])
            self.money = np.array([money], dtype='int')
            
            # Add manifest to client
            manifest = action[len(self.buyContingent):]
            for i, client in enumerate(self.clients):
                if client.alive and client.needsManifest:
                    client.setManifest(manifest)
                    break
                    
            allClientsHaveManifest = all(client.alive and not client.needsManifest for client in self.clients)
            if allClientsHaveManifest:
                for cdn in self.edgeServers:
                    cdn.manageClients(time)
                time += 1

            self.time = np.array([time], dtype='int')
        
        observation = self._get_obs()
        info = self._get_info(reward)
        self.render()

        return observation, reward, terminated, False, info

    def render(self, mode="human"):
        pass
        

if __name__ == "__main__":
    print('### Start ###')
    env = GymSabreEnv(render_mode="human", clients=2, serviceLocations=2, saveData=True, contentSteering=False)
    env = RecordEpisodeStatistics(env)
    observation, info = env.reset()

    for i in range(7_200):
        progress = round(i / 7200 * 100,0)
        #print('Progress:', progress, '/100')

        action = env.action_space.sample() # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            quit()

        # if reward != 0: 
        #     print('step:', i)
        #     print(reward) 

    env.close()
    print('### Done ###')
