'''
In this environment the CP-Agent goals is the provide all clients with a good QoE. For doing so it needs to buy contigents 
from CDNs and create manifests for clients. The reward is the cumulative reward of all clients and the money left at the CP-Agent.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces, utils

from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from datetime import datetime

from sabreEnv.gymSabre.client import Client
from sabreEnv.gymSabre.cdn import EdgeServer
from sabreEnv.gymSabre.util import Util
#from sabreEnv.gymSabre.render import run_dash_app

import pandas as pd

gym.logger.set_level(10) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridWidth=100, gridHeight=100, cdnLocations=4, \
                 maxActiveClients=10, totalClients=100, saveData=False, contentSteering=False, \
                ttl=500, maxSteps=1000, manifestLenght=4):
        # Util
        self.util = Util()
        
        # For recordings
        self.saveData = saveData
        self.episodeCounter = 0

        # For rendering
        if render_mode == 'human':
            renderData = {
                'episode': [],
                'step': [],
                'id': [],
                'type': [],
                'x': [], 
                'y': [],
                'x_target': [],
                'y_target': [],
                'alive': []
            }
            self.renderData = pd.DataFrame(renderData)

        # Env variables
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.gridSize = gridWidth * gridHeight
        self.maxSteps = maxSteps
                
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
                'clientLocation': gym.spaces.Discrete(self.gridSize+1, start=0),
                'clientsLocations': gym.spaces.MultiDiscrete([self.gridSize+1] * maxActiveClients),
                'cdnLocations': gym.spaces.MultiDiscrete([self.gridSize] * cdnLocations),
                'cdnPrices': spaces.Box(0, 10, shape=(cdnLocations,), dtype=float),
                'time': spaces.Box(0, 100_000, shape=(1,), dtype=int),
                'money': spaces.Box(0, 100_000, shape=(1,), dtype=int),
            }
        )
 
        # Action space for CP agent. Contains buy contigent and manifest for clients.
        self.buyContingent = [100] * cdnLocations
        self.manifest = manifestLenght * [cdnLocations]
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.manifest)

    def _get_obs(self):
        # Client who receices a manifest. If no client needs a manifest, the clientManifest is the gridSize.
        clientManifest = self.gridSize
        for client in self.clients:
            if client.alive and client.needsManifest:
                clientManifest = client.location

        # clientsLocations
        clientsLocations = []
        for client in self.clients:
            clientsLocations.append(client.location)

        # Fill clientsLocations with gridSize so that observation space doesn't change
        while len(clientsLocations) < self.maxActiveClients:
            clientsLocations.append(self.gridSize)

        return {
            'clientLocation': np.array(clientManifest), 
            'clientsLocations': clientsLocations, 
            'cdnLocations': self.cdnLocations, 
            'cdnPrices': self.cdnPrices, 
            'time': self.time, 
            'money': self.money
        }

    def _get_info(self, reward):
        self.sumReward += reward
        return {'money': self.money, 'time': self.time, 'sumReward': self.sumReward, \
                'activeClients': self.clients, 'cdnLocations': self.cdnLocations, 'cdnPrices': self.cdnPrices}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)# We need the following line to seed self.np_random

        # For recordings
        self.filename = 'data/gymSabre_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json'
        self.data = []
        self.util.episodeCounter += 1
        self.stepCounter = 0
        self.episodeCounter += 1
        
        # Reset env variables
        self.sumReward = 0
        self.time = np.array([0], dtype='int')

        # Reset CP-Agent
        self.money = np.array([0], dtype='int')

        # Reset CDN servers
        self.cdnLocations = self.np_random.integers(0, self.gridSize, size=self.cdnCount, dtype=int)
        self.cdnPrices = np.round(np.random.uniform(0.5, 2, size=self.cdnCount), 2)
        self.cdns = []
        for e in range(self.cdnCount):
            self.cdns.append(EdgeServer(self.util, e, self.cdnLocations[e].item(), self.cdnPrices[e]))

        # Reset clients
        self.clients = []
        self.clientIDs = 0
        self.totalClients = self.totalClientsReset

        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        self.stepCounter += 1

        time = self.time.item()
        money = self.money.item()
        reward = 0
        setManifest = False # Flag to check if a manifest had been set
        spendMoney = 0

        # Add clients
        self.clientAdder(time)

        # An episode is done when the CP is out of money, the last step is reached, or when all clients are done.
        allClientsDone = all(not client.alive for client in self.clients) and self.totalClients <= 0
        if allClientsDone:
            print('All clients done.')
        elif money <= -1_000_000:
            print('Money is below -1_000_000.')
        elif self.stepCounter >= self.maxSteps:
            print('Maximal step is reached.')
        terminated = money <= -1_000_000 or self.stepCounter >= self.maxSteps or allClientsDone

        if terminated:
            pass

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
            # Add manifest to client
            manifest = action[len(self.buyContingent):]
            for i, client in enumerate(self.clients):
                if client.alive and client.needsManifest:
                    client.setManifest(manifest.tolist())
                    setManifest = True
                    break
            
            allClientsHaveManifest = all(client.alive and not client.needsManifest for client in self.clients)
            if allClientsHaveManifest and not setManifest:
                # Buy contigent from CDNs
                # buyContigent = action[:len(self.buyContingent)]
                # for i, cdn in enumerate(self.cdns):
                #     if hasattr(buyContigent[i], 'item'):
                #         money = cdn.sellContigent(money, buyContigent[i].item())
                #     else:
                #         money = cdn.sellContigent(money, buyContigent[i])
                
                for cdn in self.cdns:
                    spendMoney += cdn.distributeNetworkConditions(time)

                # Let client do its move
                metrics = {}
                for client in self.clients:
                    metrics[client.id] = client.step(time)
                    pass
                
                # Collect information for reward                
                reward = self.reward(metrics, spendMoney)

                time += 1
                
            self.money = np.array([money], dtype='int')
            self.time = np.array([time], dtype='int')
        
        observation = self._get_obs()
        info = self._get_info(reward)
        self.render()

        return observation, reward, terminated, False, info
    
    def reward(self, metrics, money):
        '''
        Possible returns from Sabre:
        - missingTrace: Sabre does not have enough trace to complete a download.
        - downloadedSegment: Sabre has downloaded a segment.
        - completed: Sabre has downloaded all segments.
        - abortedStreaming: Sabre has aborted streaming.
        - delay: Sabre has a delay, because it buffered enough content already.
        '''
        if len(metrics) == 0: return 0
        reward = 0
        for metric in metrics.values():
            if metric['status'] == 'completed' or metric['status'] == 'downloadedSegment':
                reward += 1
            elif metric['status'] == 'abortStreaming':
                reward -= 10
        
        return (reward / len(metrics)) + money

    def render(self, mode="human"):
        if mode == "human":

            # Collect data
            for cdn in self.cdns:
                id = cdn.id
                x,y = self.get_coordinates(cdn.location, self.gridSize)
                newRow = {'episode': self.episodeCounter, 'step': self.stepCounter, 'id': id, 'type': 'CDN', \
                          'x': x, 'y': y, 'x_target': 0, 'y_target': 0, 'alive': True}
                self.renderData = pd.concat([self.renderData, pd.DataFrame([newRow])], ignore_index=True)

            for c in self.clients:
                id = c.id
                x,y = self.get_coordinates(c.location, self.gridSize)
                if c.cdn is None:
                    x_target,y_target = x,y
                else:
                    x_target,y_target = self.get_coordinates(c.cdn.location, self.gridSize)
                newRow = {'episode': self.episodeCounter, 'step': self.stepCounter, 'id':id, 'type': 'Client', \
                          'x': x, 'y': y, 'x_target': x_target, 'y_target': y_target, 'alive': c.alive}
                self.renderData = pd.concat([self.renderData, pd.DataFrame([newRow])], ignore_index=True)

    def close(self):
        super().close()
        if self.saveData:
            self.renderData.to_csv('sabreEnv/gymSabre/data/renderData.csv', index=False)
            gym.logger.info('Data saved to renderData.csv')
        
    def clientAdder(self, time):
        '''
        Here the appearing of clients is managed. 
        '''
        if self.totalClients <= 0:
            pass
        elif len(self.clients) >= self.maxActiveClients:
            pass
        elif len(self.clients) < self.maxActiveClients:
            if self.np_random.integers(1, 10) > 3 or len(self.clients) <= 0:
                c = Client(self.clientIDs, self.np_random.integers(0, self.gridSize), self.cdns, util=self.util, \
                           contentSteering=self.contentSteering, ttl=self.ttl)
                self.clientIDs += 1
                self.totalClients -= 1
                self.clients.append(c)

    def get_coordinates(self, single_integer, grid_width):
        '''
        Calculates x and y coordinates from a single integer.
        '''
        grid_width = 100 # TODO: Remove this hard coded value
        x = single_integer % grid_width
        y = single_integer // grid_width
        return x, y

if __name__ == "__main__":
    print('### Start ###')
    steps = 1_000

    env = GymSabreEnv(render_mode="human", maxActiveClients=5, totalClients=100, cdnLocations=10, saveData=True, contentSteering=True, maxSteps=steps)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=steps)
    observation, info = env.reset()

    for i in range(steps):
        progress = round(i / steps * 100,0)
        print('Progress:', progress, '/100')

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            env.close()
            exit()

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
