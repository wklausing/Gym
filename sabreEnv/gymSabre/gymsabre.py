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

gym.logger.set_level(10) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridSize=100*100, cdnLocations=4, \
                 maxActiveClients=10, totalClients=100, saveData=False, contentSteering=False, ttl=500):
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
        self.cdnCount = cdnLocations
        self.cdnLocations = np.ones(cdnLocations, dtype=int)
        self.cdnPrices = np.ones(cdnLocations, dtype=int)

        # Client variables
        self.maxActiveClients = maxActiveClients
        self.totalClients = self.totalClientsReset = totalClients
        self.contentSteering = contentSteering
        self.ttl = ttl

        # Observation space for CP agent. Contains location of clients, location of edge-servers, pricing of edge-server, and time in seconds.        
        self.observation_space = spaces.Dict(
            {
                'client': gym.spaces.Discrete(gridSize+1, start=-1),
                'clientsLocations': gym.spaces.MultiDiscrete([gridSize] * maxActiveClients),
                'cdnLocations': gym.spaces.MultiDiscrete([gridSize] * cdnLocations),
                'cdnPrices': spaces.Box(0, 10, shape=(cdnLocations,), dtype=float),
                'time': spaces.Box(0, 100_000, shape=(1,), dtype=int),
                'money': spaces.Box(0, 100_000, shape=(1,), dtype=int),
            }
        )
 
        # Action space for CP agent. Contains buy contigent and manifest for clients.
        self.buyContingent = [100] * cdnLocations
        self.manifest = cdnLocations * [cdnLocations]
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.manifest)

    def _get_obs(self):
        # Client who receices a manifest
        clientManifest = -1
        for client in self.clients:
            if client.alive and client.needsManifest:
                clientManifest = client.location

        # clientsLocations
        clientsLocations = []
        self.maxActiveClients
        for client in self.clients:
            clientsLocations.append(client.location)

        # Fill clientsLocations with -1 so that observation space doesn't change
        while len(clientsLocations) < self.maxActiveClients:
            clientsLocations.append(-1)

        return {
            'client': np.array(clientManifest), 
            'clientsLocations': clientsLocations, 'serviceLocations': self.cdnLocations
                , 'cdnPrices': self.cdnPrices, 'time': self.time
                , 'money': self.money}

    def _get_info(self, reward):
        self.sumReward += reward
        return {'money': self.money, 'time': self.time, 'sumReward': self.sumReward, \
                'activeClients': self.clients, 'cdnLocations': self.cdnLocations, 'cdnPrices': self.cdnPrices}

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
        self.cdnLocations = self.np_random.integers(0, self.gridSize, size=self.cdnCount, dtype=int)
        self.cdnPrices = np.round(np.random.uniform(0.5, 2, size=self.cdnCount), 2)
        self.cdns = []
        for e in range(self.cdnCount):
            self.cdns.append(EdgeServer(self.util, e, self.cdnLocations[e].item(), self.cdnFilename, self.cdnPrices[e]))

        # Reset clients
        self.clients = []
        self.clientIDs = 0
        self.totalClients = self.totalClientsReset

        observation = self._get_obs()
        info = {}
        
        return observation, info
    
    def clientAdder(self, time):
        '''
        Here the appearing of clients is managed. 
        '''
        self.oldTime = time
        if self.totalClients <= 0:
            pass
        elif len(self.clients) >= self.maxActiveClients:
            pass
        elif len(self.clients) < 10:
            if self.np_random.integers(1, 10) > 3 or len(self.clients) <= 0:
                c = Client(self.clientIDs, self.np_random.integers(0, self.gridSize), self.cdns, util=self.util, contentSteering=self.contentSteering, ttl=self.ttl)
                self.clientIDs += 1
                self.totalClients -= 1
                self.clients.append(c)
        else:
            pass
        pass

    def step(self, action):
        self.stepCounter += 1

        time = self.time.item()
        money = self.money.item()
        reward = 0

        # Add clients
        self.clientAdder(time)

        # An episode is done when the CP is out of money, the last step is reached, or when all clients are done.
        allClientsDone = all(not client.alive for client in self.clients) and self.totalClients <= 0
        if allClientsDone:
            print('All clients done.')
        elif money <= -1_000_000:
            print('Money is below -1_000_000.')
        elif self.stepCounter >= 7_200:
            print('Maximal step is reached.')
        terminated = money <= -1_000_000 or self.stepCounter >= 7_200 or allClientsDone

        # Saving data
        clients_to_remove = []
        for client in self.clients:
            if not client.alive or terminated:
                client.saveData(finalStep=self.saveData)
                clients_to_remove.append(client)
        for cdn in self.cdns:
            cdn.saveData(time, finalStep=terminated)

        # Remove clients
        for client in clients_to_remove:
            self.clients.remove(client)

        if terminated:
            print('Termination time:', time)
            pass
        else:
            # Buy contigent from CDNs
            buyContigent = action[:len(self.buyContingent)] 
            for i, cdn in enumerate(self.cdns):
                if hasattr(buyContigent[i], 'item'): 
                    money = cdn.sellContigent(money, buyContigent[i].item())
                else:
                    money = cdn.sellContigent(money, buyContigent[i])
            self.money = np.array([money], dtype='int')

            # Add manifest to client
            manifest = action[len(self.buyContingent):]
            for i, client in enumerate(self.clients):
                if client.alive and client.needsManifest:
                    client.setManifest(manifest.tolist())
                    break
            
            # Manage clients
            allClientsHaveManifest = all(client.alive and not client.needsManifest for client in self.clients)
            if allClientsHaveManifest:
                for cdn in self.cdns:
                    cdn.manageClients(time)
                time += 1

            # Let client do its move
            for client in self.clients:
                clientMetrics = client.step(time)
                reward += clientMetrics['qoe']
                pass

            self.time = np.array([time], dtype='int')
        
        observation = self._get_obs()
        info = self._get_info(reward)
        self.render()

        return observation, reward, terminated, False, info

    def render(self, mode="human"):
        pass
        

if __name__ == "__main__":
    print('### Start ###')
    env = GymSabreEnv(render_mode="human", maxActiveClients=10, cdnLocations=2, saveData=True, contentSteering=False)
    env = RecordEpisodeStatistics(env)
    observation, info = env.reset()

    steps = 100_000
    for i in range(steps):
        progress = round(i / steps * 100,0)
        print('Progress:', progress, '/100')

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            pass

        if i == steps-1:
            print('All steps done.')

    env.close()
    print('### Done ###')
    print(f'Episode total rewards: {env.return_queue}')
    print(f'Episode lengths: {env.length_queue}')
    print(f'Episode count: {env.episode_count}')
    print(f'Episode start time: {env.episode_start_times}')
    print(f'Episode returns: {env.episode_returns}')
    print(f'Episode lengths: {env.episode_lengths}')
