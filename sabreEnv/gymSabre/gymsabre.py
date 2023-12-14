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

import json
import os

from datetime import datetime

gym.logger.set_level(50) # Define logger level. 20 = info, 30 = warn, 40 = error, 50 = disabled

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, gridSize = 100*100, edgeServers = 4, clients = 10):

        self.filename = 'render' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json'

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

        self.data = []
        
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
            self.edgeServers.append(EdgeServer(e, self.edgeServerLocations[e].item(), self.edgeServerPrices[e]))

        # Reset clients
        self.clientsLocations = self.np_random.integers(0, self.gridSize, size=self.clientCount, dtype=int)
        self.clients = []
        for c in range(self.clientCount):
            self.clients.append(Client(c ,self.clientsLocations[c].item(), self.edgeServers))

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

        if terminated:
            name = 'render' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json'
            self.add_to_json_file(name, self.data)
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
                    if result['status'] == 'fetching':
                        reward += result['qoe']
                    elif result['status'] == 'done':
                        gym.logger.info('Client %s is done.' % client.id)
                    elif result['status'] == 'missingTrace':
                        gym.logger.info('Client %s was missing trace.' % client.id)
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
        # print('#########',self.time.item(),'##########')
        # print('#########CDNs##########')
        # for edgeServer in self.edgeServers:
        #     print('id: %s, location: %s, price: %s, contigent: %s' % (edgeServer.id, edgeServer.location, edgeServer.price, edgeServer.contigent))
        # print('#########Clients##########')
        # for client in self.clients:
        #     print('id: %s, location: %s, alive: %s, edgeServer: %s' % (client.id, client.location, client.alive, client.edgeServer.id))

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
                'contigent': edgeServer.contigent
            }
            info_dict['CDNs'].append(edge_server_info)

        # Adding Client information
        for client in self.clients:
            client_info = {
                'id': client.id,
                'location': client.location,
                'alive': client.alive,
                'edgeServer': client.edgeServer.id if client.edgeServer else None
            }
            info_dict['Clients'].append(client_info)

        self.data.append(info_dict)

        #self.add_to_json_file('render.json', info_dict)

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

        # Sabre implementation
        self.sabre = Sabre(verbose=False)   

    def setManifest(self, manifest):
        self.manifest = manifest
        self.idxManifest = 0
        self.edgeServer = self.cdns[self.manifest[self.idxManifest]]
        self.edgeServer.addClient(self)

    def fetchContent(self, time):
        '''
        If fetch origin can delivier than return 1. If not than increase idxCDN to select next server and return -1.
        Here Sabre should be used to get a real reward based on QoE. 

        The CP gets money from clients whenever they fetch content.
        '''
        # Distance between client and edge-server. Used for latency calculation.
        latency = self._determineLatency(self.location, self.edgeServer.location)

        # Bandwidth
        bandwidth = self.edgeServer.currentBandwidth

        # Each time a client fetches content it pays the edge-server.
        if self.edgeServer.contigent > 0:
            # Add network trace to client
            self.time = self.time - time
            self.sabre.network.add_network_condition(duration_ms=time*1000, bandwidth_kbps=bandwidth, latency_ms=latency)
        else:
            gym.logger.info('Not enough contingent for client %s at CDN %s.' % (self.id, self.edgeServer.id))
            self._changeCDN()
            self.sabre.network.remove_network_condition()

        qoe = self._getQoE()
        if qoe['status'] == 'fetching':
            self.edgeServer.deductContigent(qoe['size'])

        return qoe
        
    def _getQoE(self):
        '''
        Here QoE will be computed with Sabre.
        '''
        result = self.sabre.downloadSegment()
        if result['done'] == False and len(result) == 6:
            qoe = result['time_average_played_bitrate'] - result['time_average_bitrate_change'] - result['time_average_rebuffer_events']
            return {'qoe': qoe, 'size': result['size'], 'status': 'fetching'}
        elif result['done']:
            self._setDone()
            return {'status': 'done'}
        else:
            gym.logger.info('Not enough trace for client %s to fetch from CDN %s.' % (self.id, self.edgeServer.id))
            return {'status': 'missingTrace'}
        
    def _changeCDN(self):
        '''
        Change CDN if i.e. QoE is bad.
        Iterates over the manifest to select next CDN. When reaching the end it stops.
        '''
        self.idxManifest += 1
        if self.idxManifest >= len(self.manifest):
            self.idxManifest -= 1
            gym.logger.warn('Client %s is already at last CDN (id=%s) in manifest.' % (self.id, self.idxManifest))
        else:
            self.edgeServer.removeClient(self)
            self.edgeServer = self.cdns[self.manifest[self.idxManifest]]
            self.edgeServer.addClient(self)
            gym.logger.info('Client %s changed to CDN %s.' % (self.id, self.idxManifest))

    def _setDone(self):
        gym.logger.info('Client %s downloaded content successfully.' % (self.id))
        self.alive = False
        self.edgeServer.removeClient(self)

    def _determineLatency(self, position1, position2):
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
        price = self.price * amount
        if cpMoney >= price:
            cpMoney -= price
            self.contigent += amount * 1000
        return round(cpMoney, 2)
    
    def addClient(self, client):
        self.clients.append(client)
        self.currentBandwidth = self.bandwidth_kbps / len(self.clients)

    def removeClient(self, client):
        self.clients.remove(client)
        if len(self.clients) == 0: return
        self.currentBandwidth = self.bandwidth_kbps / len(self.clients)

    def deductContigent(self, amount):
        amountMB = amount / 8_000_000
        if self.contigent >= amountMB:
            self.contigent -= amountMB
            self.contigent = round(self.contigent, 2)
        else:
            gym.logger.warn('Deducting too much contigent from CDN %s.' % self.id)
            quit()
    
    @property
    def bandwidth(self):
        return self.bandwidth_kbps


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

        if reward != 0: print(reward)

    env.close()
    print('### Done ###')
