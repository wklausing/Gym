'''
In this environment the CP-Agent goals is the provide all clients with a good QoE. For doing so it needs to buy contigents 
from CDNs and create manifests for clients. The reward is the cumulative reward of all clients and the money left at the CP-Agent.
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from gymnasium.wrappers import RecordEpisodeStatistics

import json
import os

from datetime import datetime

from sabreEnv.gymSabre.client import Client
from sabreEnv.gymSabre.cdn import EdgeServer

gym.logger.set_level(50) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridSize = 100*100, edgeServers = 4, clients = 10, saveData = False):

        # For recordings
        self.saveData = saveData
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clientFilename = 'data/client_' + time + '.csv'
        self.cdnFilename = 'data/cdn_' + time + '.csv'
        self.cpFilename = 'data/cp_' + time + '.csv'
        self.episodeCounter = 0

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
        self.manifest = clients * edgeServers * [edgeServers] * 2
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.manifest)

    def _get_obs(self):
        #clientsLocations
        for client in self.clients:
            id = client.id
            self.clientsLocations[id] = self.clients[id].location

        return {'clientsLocations': self.clientsLocations, 'edgeServerLocations': self.edgeServerLocations
                , 'edgeServerPrices': self.edgeServerPrices, 'time': self.time
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
        self.episodeCounter += 1
        
        # Reset env variables
        self.sumReward = 0
        self.time = np.array([0], dtype='int')

        # Reset CP-Agent
        self.money = np.array([100], dtype='int')

        # Reset edge servers
        self.edgeServerLocations = self.np_random.integers(0, self.gridSize, size=self.edgeServerCount, dtype=int)
        self.edgeServerPrices = np.round(np.random.uniform(0.5, 2, size=self.edgeServerCount), 2)
        self.edgeServers = []
        for e in range(self.edgeServerCount):
            self.edgeServers.append(EdgeServer(e, self.edgeServerLocations[e].item(), self.cdnFilename, self.episodeCounter, self.edgeServerPrices[e]))

        # Reset clients
        self.clientsLocations = self.np_random.integers(0, self.gridSize, size=self.clientCount, dtype=int)
        self.clients = []
        for c in range(self.clientCount):
            self.clients.append(Client(c ,self.clientsLocations[c].item(), self.edgeServers, self.clientFilename, self.episodeCounter))

        observation = self._get_obs()
        info = {}
        
        return observation, info

    def step(self, action):
        time = self.time.item()
        money = self.money.item()
        reward = 0

        # An episode is done when the CP is out of money or time is over
        allClientsDone = all(not client.alive for client in self.clients)
        terminated = money <= 0 or time >= 7_200 or allClientsDone

        # Saving data
        if self.saveData:
            for client in self.clients:
                if client.alive:
                    client.saveData(finalStep=terminated)
            for edgeServer in self.edgeServers:
                edgeServer.saveData(time, finalStep=terminated)

        if terminated:
            pass
        else:
            # Buy contigent
            buyContigent = action[:len(self.buyContingent)] 
            for i, edgeServer in enumerate(self.edgeServers):
                if hasattr(buyContigent[i], 'item'): 
                    money = edgeServer.sellContigent(money, buyContigent[i].item())
                else:
                    money = edgeServer.sellContigent(money, buyContigent[i])
            
            # Create manifest and let client fetch content
            manifest = action[len(self.buyContingent):] 
            for i, client in enumerate(self.clients):
                if client.alive:
                    if not hasattr(client, 'manifest'):
                        start = client.id*4
                        m = manifest[start:4+start]
                        client.setManifest(m)
                    result = client.fetchContent(time)
                    if result['status'] == 'downloadedSegment' or result['status'] == 'completed':
                        reward += result['qoe']
                    elif result['status'] == 'missingTrace':
                        gym.logger.info('Client %s was missing trace.' % client.id)
                    elif result['status'] == 'delay':
                        gym.logger.info('Client has delay of %s.' % result['delay'])
                    else:
                        gym.logger.info('Client %s could not fetch content from CND %s.' % (client.id, client.edgeServer.id))                 

        time += 1
        self.render()

        self.time = np.array([time], dtype='int')
        self.money = np.array([money], dtype='int')
        
        observation = self._get_obs()
        info = self._get_info(reward)

        return observation, reward, terminated, False, info
    
    data = []

    def render(self, mode="human"):

        info_dict = {
            'time': self.time.item(),
            'CDNs': [],
            'Clients': []
        }

        # Adding CDN information
        for edgeServer in self.edgeServers:
            edge_server_info = {
                'id': edgeServer.id,
                'location': edgeServer.location,
                'price': edgeServer.price,
                'contigent': edgeServer.contigent,
                'servingClients': [client.id for client in edgeServer.clients],
                'money': edgeServer.money
            }
            info_dict['CDNs'].append(edge_server_info)

        # Adding Client information
        for client in self.clients:
            client_info = {
                'id': client.id,
                'location': client.location,
                'alive': client.alive,
                'edgeServer': client.edgeServer.id if client.edgeServer else None,
                'qoe': client.qoe
            }
            info_dict['Clients'].append(client_info)

        self.data.append(info_dict)

    def add_to_json_file(self, filename, new_data):
        """
        Adds a new entry to a JSON file. Creates the file with the initial data if it doesn't exist.

        :param filename: The name of the JSON file.
        :param new_data: The new data entry (a dictionary) to be added.
        """
        # Check if file exists
        if os.path.isfile(self.filename):
            # Read existing data
            with open(self.filename, 'r') as file:
                data = json.load(file)
        else:
            # Create an empty list if file does not exist
            data = []

        # Add new data entry
        data.append(new_data)

        # Write data back to file
        with open(self.filename, 'w') as file:
            json.dump(data, file, indent=4)


if __name__ == "__main__":
    print('### Start ###')
    env = GymSabreEnv(render_mode="human", clients=10, edgeServers=4)
    env = RecordEpisodeStatistics(env)
    observation, info = env.reset()

    for i in range(7_200):
        progress = round(i / 7200 * 100,0)
        print('Progress:', progress, '/100')

        action = env.action_space.sample() # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        if reward != 0: 
            print('step:', i)
            print(reward)

    env.close()
    print('### Done ###')
