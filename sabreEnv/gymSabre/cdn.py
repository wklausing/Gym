import gymnasium as gym
import numpy as np

import math

class CDN:

    def __init__(self, util, id, location, price=1.0, bandwidth_kbps=10000, reliable=100, random=np.random):
        self.util = util

        self.id = id
        self.location = location
        self.price = price
        self.contigent = 0 # In bits
        self.bandwidth_kbps = bandwidth_kbps
        self.currentBandwidth = bandwidth_kbps # Bandwidth for each client
        self.clients = []
        self.money = 0

        self.reliable = reliable
        self.random = random

        self.buffered_data = []
    
    def addClient(self, client):
        self.clients.append(client)
        self.currentBandwidth = round(self.bandwidth_kbps / len(self.clients),2)

    def removeClient(self, client):
        self.clients.remove(client)
        if len(self.clients) == 0: return
        self.currentBandwidth = round(self.bandwidth_kbps / len(self.clients),2)

    def sellContigent(self, cpMoney, amount):
        '''
        Input amount is in GB.
        '''
        price = round(self.price * amount,2)
        cpMoney -= price
        self.money += price
        self.contigent += amount * 8_000_000_000
        gym.logger.info('CP bought %s GBs from CDN %s for a price of %s.' % (amount, self.id, self.price))
        return round(cpMoney, 2)    

    def deductContigent(self, duration, bandwidth, latency):
        '''
        Deduct amount of contigent from CDN.
        '''
        amount = bandwidth * duration
        if self.contigent >= amount:
            self.contigent -= amount
            self.contigent = round(self.contigent, 2)            
        else:
            gym.logger.warn('Deducting too much contigent from CDN %s.' % self.id)
            return False
        return True
    
    @property
    def bandwidth(self):
        pass
        return self.bandwidth_kbps
    
    @property
    def normBandwidth(self):
        return self.currentBandwidth / self.bandwidth_kbps
    
    clientsStatus = {}
    def distributeNetworkConditions(self, time):
        '''
        Here clients receive network conditions.
        '''
        
        # Add reliability feature of CDN
        random = self.random.integers(1, 100)
        if random > self.reliable:
            gym.logger.info('CDN %s has a problem, bandwidth is halved.' % self.id)
            self.bandwidth_kbps = self.bandwidth_kbps / 2

        # Distribute bandwidth
        clients = [client for client in self.clients if client.status in ['missingTrace', 'downloadedSegment', 'init']]
        for client in clients:
            duration = 1000
            bandwidth = self.bandwidth_kbps / len(clients)
            latency = self._determineLatency(self.location, client.location)
            client.provideNetworkCondition(duration_ms=duration, bandwidth_kbps=bandwidth, latency_ms=latency)
        if len(clients) > 0:
            gym.logger.info('CP pays %s for using CDN %s.' % (self.price, self.id))
            return self.price
        else:
            return 0

    def _determineLatency(self, position1, position2):
        '''
        Calculates distance between two points. Used to calcualte latency.
        '''
        distance = self.util.calcDistance(position1, position2)
        distance = round(distance * 5, 2)
        if distance < 1: distance = 1
        return distance

    def saveData(self, time, finalStep=False):
        if time == 0 or self.util.saveData == False: return
            
        client_ids = [client.id for client in self.clients]
        self.buffered_data.append([self.util.episodeCounter, time, self.id, len(client_ids), np.array(client_ids), self.bandwidth_kbps, \
                                   self.currentBandwidth, self.price, self.money, self.contigent])
        if finalStep:
            self.util.cdnCsvExport(self.buffered_data)
            gym.logger.info('SaveData for cdn %s.' % self.id)