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
from sabreEnv.gymSabre.cdn import CDN
from sabreEnv.gymSabre.util import Util
#from sabreEnv.gymSabre.render import run_dash_app

import math

import pandas as pd

gym.logger.set_level(30) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridWidth=100, gridHeight=100, \
                    cdns=4, cdnLocationsFixed=[3333, 3366, 6633, 6666], cdnBandwidth=1000, cdnReliable=[100], shuffelPrice=999999999999, \
                    maxActiveClients=10, totalClients=100, clientAppearingMode='constante', manifestLenght=4, \
                    bufferSize=25, mpdPath='sabreEnv/sabre/data/movie_30s.json', \
                    contentSteering=False, ttl=500, maxSteps=1_000, \
                    saveData=False, savingPath='sabreEnv/gymSabre/data/', filePrefix='D', \
                    weightQoE=2, weightCost=1, weightAbort=1, dqnActionSpace=True
                ):
        
        # Checking input parameters
        assert gridWidth > 0, 'gridWidth must be greater than 0.'
        assert gridHeight > 0, 'gridWidth must be greater than 0.'
        assert cdns > 1, 'cdns must be greater than 1.'
        assert len(cdnLocationsFixed) <= cdns, 'cdnLocationsFixed must be smaller or equal to cdns.'
        assert cdnBandwidth > 0, 'cdnBandwidth must be greater than 0.'
        assert all(0 <= value <= 100 for value in cdnReliable) or not cdnReliable, "List must be empty or contain values between 0 and 100"
        assert maxActiveClients > 0, 'maxActiveClients must be greater than 0.'
        assert totalClients > 0, 'totalClients must be greater than 0.'
        assert manifestLenght > 0, 'manifestLenght must be greater than 0.'
        assert maxActiveClients <= totalClients, 'totalClients must be greater or equal to maxActiveClients.'
        assert ttl >= 0, 'ttl must be greater or equal than 0.'
        assert maxSteps > 0, 'maxSteps must be greater than 0.'
        assert bufferSize > 0, 'bufferSize must be greater than 0.'

        # Util
        self.util = Util(saveData=saveData, savingPath=savingPath, filePrefix=filePrefix, gridWidth=gridWidth, gridHeight=gridHeight)
        self.filePrefix = filePrefix
        self.savingPath = savingPath
        
        # For recordings
        self.saveData = saveData
        self.episodeCounter = 0

        # For rendering
        self.render_mode = render_mode            
        if self.saveData:
            self.renderData = pd.DataFrame(columns=['episode', 'step', 'id', 'x', 'y', 'x_target', 'y_target', 'alive'])
            self.cpData = pd.DataFrame(columns=['episode', 'time', 'reward', 'qoe', 'costsNorm', 'costTotal', 'step'])

        # Env variables
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.gridSize = gridWidth * gridHeight
        self.maxSteps = maxSteps
        self.clientAppearingMode = clientAppearingMode
        self.weightQoE = weightQoE
        self.weightCost = weightCost
        self.weightAbort = weightAbort
        self.dqnActionSpace = dqnActionSpace
                
        # CDN variables
        self.cdnCount = cdns
        self.cdnLocationsFixed = cdnLocationsFixed
        self.cdnPrices = np.ones(cdns, dtype=int)
        self.cdnBandwidth = cdnBandwidth
        self.cdnReliable = cdnReliable
        self.shuffelPrice = shuffelPrice

        # Client variables
        self.maxActiveClients = maxActiveClients
        self.totalClients = self.totalClientsReset = totalClients
        self.contentSteering = contentSteering
        self.ttl = ttl
        self.bufferSize = bufferSize
        self.mpdPath = mpdPath

        # Observation space for CP agent. Contains location of clients, location of edge-servers, pricing of edge-server, and time in seconds.        
        self.observation_space = spaces.Dict(
            {
                'clientLocation': gym.spaces.Discrete(self.gridSize+1, start=0),
                'clientsLocations': gym.spaces.MultiDiscrete([self.gridSize+1] * maxActiveClients),
                'cdnLocations': gym.spaces.MultiDiscrete([self.gridSize] * cdns),
                'cdnPrices': spaces.Box(0, 10, shape=(cdns,), dtype=float),
                'time': spaces.Box(0, 100_000, shape=(1,), dtype=int)
            }
        )
 
        # Action space for CP agent. Contains buy contigent and manifest for clients.
        if dqnActionSpace == True:
            self.action_space = gym.spaces.Discrete(int(pow(cdns, manifestLenght)))
        else:
            self.action_space = gym.spaces.MultiDiscrete(manifestLenght * [cdns])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # For recordings
        self.data = []
        self.stepCounter = 0
        self.episodeCounter += 1
        self.util.episodeCounter = self.episodeCounter

        # Reset env variables
        self.sumReward = 0
        self.time = np.array([0], dtype='int')
        self.newTime = True
        self.enterClientAdderFirstTime = True
        self.setManifestCount = 0

        # Reset CP-Agent
        self.money = np.array([0], dtype='int')

        # Reset CDN servers
        self.cdnLocationsFixed = np.array(self.cdnLocationsFixed)
        filtered_fixed_locations = [loc for loc in self.cdnLocationsFixed if 0 <= loc < self.gridSize]
        remaining_count = self.cdnCount - len(filtered_fixed_locations)
        if remaining_count > 0:
            random_locations = self.np_random.integers(0, self.gridSize, size=remaining_count, dtype=int)
            self.cdnLocations = filtered_fixed_locations + list(random_locations)
        else:
            self.cdnLocations = filtered_fixed_locations

        #self.cdnLocations = self.np_random.integers(0, self.gridSize, size=self.cdnCount, dtype=int)
        self.cdnPrices = np.round(self.np_random.uniform(0.02, 0.07, size=self.cdnCount), 2)
        self.cdns = []
        for e in range(self.cdnCount):
            reliable = self.cdnReliable[e] if e < len(self.cdnReliable) and e >= 0 else 100
            self.cdns.append(CDN(self.util, e, self.cdnLocations[e].item(), self.cdnPrices[e], bandwidth_kbps=self.cdnBandwidth, reliable=reliable, random=self.np_random))

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
        metrics = {}

        # Add clients
        self._clientAdder(time, mode=self.clientAppearingMode)
        self.newTime = False

        # Shuffel prices
        if self.time > 1 and self.time % self.shuffelPrice == 0:
            oldPrices = self.cdnPrices
            self.cdnPrices = np.round(self.np_random.uniform(0.02, 0.07, size=self.cdnCount), 2)
            for i, cdn in enumerate(self.cdns):
                cdn.price = round(self.cdnPrices[i], 2)
            gym.logger.info(f'Prices shuffeld: {oldPrices} -> {self.cdnPrices}')

        # An episode is done when the CP is out of money, the last step is reached, or when all clients are done.
        allClientsDone = all(not client.alive for client in self.clients) and self.totalClients <= 0
        if allClientsDone:
            gym.logger.info('All clients done.')
        elif self.stepCounter >= self.maxSteps:
            gym.logger.info('Maximal step is reached.')
        terminated = self.stepCounter >= self.maxSteps or allClientsDone

        if terminated:
            pass

        # Saving data
        clients_to_remove = []
        #if self.saveData:            
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
            print('Termination time:', time, ' Step:', self.stepCounter)
            pass
        else:
            # Add manifest to client
            if self.dqnActionSpace:
                manifest = self.interpret_action(action, 4, self.cdnCount)
            else:
                manifest = action
            for _, client in enumerate(self.clients):
                if client.alive and client.needsManifest:
                    client.setManifest(manifest)
                    setManifest = True
                    self.setManifestCount += 1
                    break
            
            allClientsHaveManifest = all(client.alive and not client.needsManifest for client in self.clients)

            if allClientsHaveManifest and not setManifest:
                
                for cdn in self.cdns:
                    spendMoney += cdn.distributeNetworkConditions(time)

                # Let client do its move
                for client in self.clients:
                    metrics[client.id] = client.step(time)
                    pass

                time += 1
                self.newTime = True
            
            self.money = np.array([money], dtype='int')
            self.time = np.array([time], dtype='int')
        
        # Collect information for reward     
        reward = self.reward(metrics, spendMoney)
        observation = self._get_obs()
        info = self._get_info(reward)
        self.render()

        gym.logger.info(f'Time: {time}' + f' Step: {self.stepCounter}')
        return observation, reward, terminated, False, info
    
    def reward(self, metrics, cost):
        '''
        Possible returns from Sabre:
        - missingTrace: Sabre does not have enough trace to complete a download.
        - downloadedSegment: Sabre has downloaded a segment.
        - completed: Sabre has downloaded all segments.
        - abortedStreaming: Sabre has aborted streaming.
        - delay: Sabre has a delay, because it buffered enough content already.
        '''
        if len(metrics) == 0: return 0
        qoe = 0
        qoeCount = 0
        abortPenalty = 0

        for metric in metrics.values():
            if metric['status'] in ['completed', 'downloadedSegment']:
                qoe += metric['normalized_qoe']
                qoeCount += 1
                if metric['normalized_qoe'] > 1: 
                    pass
            elif metric['status'] == 'abortedStreaming':
                abortPenalty += 3

        costsNorm = -1
        costsNorm += self._determineNormalizedPrices(self.cdnPrices, cost)

        qoe = qoe / qoeCount if qoeCount > 0 else 0
        reward = qoe * self.weightQoE - cost * self.weightCost - abortPenalty * self.weightAbort

        # Collect data for CP graphs
        if self.saveData:
            newRow = {'episode': self.episodeCounter, 
                      'time': self.time.item(), 
                      'reward': reward, 
                      'qoe': qoe, 
                      'costsNorm': costsNorm,
                      'costTotal': cost,
                      'step': self.stepCounter}
            if cost <= 0:
                pass
            self.cpData = pd.concat([self.cpData, pd.DataFrame([newRow])], ignore_index=True)

        return reward

    def render(self, mode="human"):
        if self.saveData:
            # Collect data
            for cdn in self.cdns:
                id = cdn.id
                x,y = self._get_coordinates(cdn.location, self.gridWidth)
                newRow = {'episode': self.episodeCounter, 'step': self.stepCounter, 'id': id, 'type': 'CDN', \
                          'x': x, 'y': y, 'x_target': 0, 'y_target': 0, 'alive': True}
                self.renderData = pd.concat([self.renderData, pd.DataFrame([newRow])], ignore_index=True)

            for c in self.clients:
                id = c.id
                x,y = self._get_coordinates(c.location, self.gridWidth)
                if c.cdn is None:
                    x_target,y_target = x,y
                else:
                    x_target,y_target = self._get_coordinates(c.cdn.location, self.gridWidth)
                newRow = {'episode': self.episodeCounter, 'step': self.stepCounter, 'id':id, 'type': 'Client', \
                          'x': x, 'y': y, 'x_target': x_target, 'y_target': y_target, 'alive': c.alive}
                self.renderData = pd.concat([self.renderData, pd.DataFrame([newRow])], ignore_index=True)

    def close(self):
        super().close()
        if self.saveData:
            self.filePrefix
            path = self.savingPath + self.filePrefix
            self.renderData.to_csv(path + 'renderData.csv', index=False)
            self.cpData.to_csv(path + 'cpData.csv', index=False)
            gym.logger.info('Data saved to renderData.csv')

    def _get_obs(self):
        # Client who receices a manifest. If no client needs a manifest, the clientManifest is the gridSize.
        self.obsClientLocation = self.gridSize
        for client in self.clients:
            if client.alive and client.needsManifest:
                self.obsClientLocation = client.location

        # clientsLocations
        clientsLocations = []
        for client in self.clients:
            clientsLocations.append(client.location)

        # Fill clientsLocations with gridSize so that observation space doesn't change
        while len(clientsLocations) < self.maxActiveClients:
            clientsLocations.append(self.gridSize)

        return {
            'clientLocation': np.array(self.obsClientLocation), 
            'clientsLocations': clientsLocations, 
            'cdnLocations': self.cdnLocations, 
            'cdnPrices': self.cdnPrices, 
            'time': self.time
        }

    def _get_info(self, reward):
        self.sumReward += reward
        return {'money': self.money, 'time': self.time, 'sumReward': self.sumReward, \
                'activeClients': self.clients, 'cdnLocations': self.cdnLocations, 'cdnPrices': self.cdnPrices}

    def _clientAdder(self, time, mode='random'):
        '''
        Here the appearing of clients is managed. Currently, it is a random process, parabolic, or exponentially.
        '''
        if self.totalClients <= 0 or not self.newTime:
            return
        
        if mode == 'constante':
            if len(self.clients) < self.maxActiveClients and self.totalClients > 0:
                while len(self.clients) < self.maxActiveClients and self.totalClients > 0:
                    self._addClient()
        elif mode == 'random':
            maxClients = self.maxActiveClients-len(self.clients)
            if self.enterClientAdderFirstTime:
                self.randomClientCount = self.np_random.integers(2, maxClients, dtype=int)
                self.randomClientMinCount = self.np_random.integers(1, self.randomClientCount, dtype=int)
                for _ in range(self.randomClientCount):
                    self._addClient()
                self.enterClientAdderFirstTime = False
            elif self.randomClientMinCount >= len(self.clients):
                if maxClients > 2:
                    self.randomClientCount = self.np_random.integers(2, maxClients, dtype=int)
                else:
                    self.randomClientCount = 2
                self.randomClientMinCount = self.np_random.integers(1, self.randomClientCount, dtype=int)
                for _ in range(self.randomClientCount):
                    self._addClient()
        elif mode == 'parabolic':
            # Calculate the number of clients to add based on a parabolic function
            midpoint = self.maxSteps / 2
            a = -4 * self.maxActiveClients / (midpoint ** 2)  # Adjust 'a' to control the peak
            num_new_clients = int(a * (time - midpoint) ** 2 + self.maxActiveClients)

            # Add clients up to the calculated number or until the maximum is reached
            for _ in range(min(num_new_clients, self.maxActiveClients - len(self.clients))):
                self._addClient()
        elif mode == 'exponentially':
            # Exponential growth parameters
            a = max(1, len(self.clients))  # Ensure a is at least 1
            b = math.log(self.maxActiveClients / a) / self.maxSteps

            num_new_clients = int(a * math.exp(b * time))
            max_addable_clients = self.maxActiveClients - len(self.clients)

            for _ in range(min(num_new_clients, max_addable_clients)):
                self._addClient()

    def _addClient(self):
        gym.logger.info('Add client %s', self.clientIDs)
        if self.totalClients <= 0: return
        c = Client(self.clientIDs, self.np_random.integers(0, self.gridSize), self.cdns, util=self.util, \
                        contentSteering=self.contentSteering, ttl=self.ttl, bufferSize=self.bufferSize, \
                              maxActiveClients=self.maxActiveClients, mpdPath=self.mpdPath)
        self.clientIDs += 1
        self.totalClients -= 1
        self.clients.append(c)

    def _get_coordinates(self, single_integer, gridWidth):
        '''
        Calculates x and y coordinates from a single integer.
        '''
        x = single_integer % gridWidth
        y = single_integer // gridWidth
        return x, y

    def _determineNormalizedPrices(self, cdnPrices, price):
        '''
        Determines the normalized prices of the CDNs. Used for reward function.
        '''
        max_value = sum(cdnPrices)
        min_value = min(cdnPrices)
        normalizedPrice = (price - min_value) / (max_value - min_value)
        return normalizedPrice
    
    def interpret_action(self, discrete_action, manifestLength, cdns):
        """
        Converts an action from the Discrete space into a list of actions in the MultiDiscrete space.

        :param discrete_action: The action in the Discrete space.
        :param manifestLength: The number of actions in the MultiDiscrete space.
        :param cdns: The number of possible values for each action in the MultiDiscrete space.
        :return: A list of actions in the MultiDiscrete space.
        """
        multi_discrete_action = []
        for _ in range(manifestLength):
            action = discrete_action % cdns
            multi_discrete_action.append(action)
            discrete_action = discrete_action // cdns

        return multi_discrete_action[::-1]
    

if __name__ == "__main__":
    print('### Start ###')
    steps = 1_000

    env = GymSabreEnv(render_mode="human", maxActiveClients=10, totalClients=100, saveData=True, contentSteering=False, ttl=10, maxSteps=steps)
    env = RecordEpisodeStatistics(env)
    env = TimeLimit(env, max_episode_steps=steps)
    observation, info = env.reset()

    for i in range(steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print('### Done ###')
    print(f'Episode total rewards: {env.return_queue}')
    print(f'Episode lengths: {env.length_queue}')
    print(f'Episode count: {env.episode_count}')
    print(f'Episode start time: {env.episode_start_times}')
    print(f'Episode returns: {env.episode_returns}')
    print(f'Episode lengths: {env.episode_lengths}')
