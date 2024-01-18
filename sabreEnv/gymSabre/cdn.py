import gymnasium as gym
import numpy as np

import math

class EdgeServer:

    def __init__(self, util, id, location, price=1, bandwidth_kbps=10000):
        self.util = util

        self.id = id
        self.location = location
        self.price = price
        self.contigent = 0# In bits
        self.bandwidth_kbps = bandwidth_kbps
        self.currentBandwidth = bandwidth_kbps # Bandwidth for each client
        self.clients = []
        self.money = 0

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
        return self.bandwidth_kbps
    
    clientsStatus = {}
    def manageClients(self, time):
        '''
        Here clients will fetch content from CDN and return the result as a dictionary.
        '''

        # Check what is needed for clients to proceed
        for client in self.clients:
            if client.status in ['missingTrace', 'downloadedSegment', 'init', 'abortedStreaming']:
                pass
            elif client.status in ['completed', 'delay']:
                self.clientsRequiresTrace =- 1
            else:
                gym.logger.warn('Unknown status %s for client %s.' % (client.status, client.id))
                quit()

        # Distribute bandwidth
        clients = [client for client in self.clients if client.status in ['missingTrace', 'downloadedSegment']]
        for client in clients:
            duration = 1000
            bandwidth = self.bandwidth_kbps / len(clients)
            latency = self._determineLatency(self.location, client.location)
            if self.deductContigent(duration, bandwidth, latency):
                client.provideNetworkCondition(duration_ms=duration, bandwidth_kbps=bandwidth, latency_ms=latency)
            else:
                client.provideNetworkCondition(duration_ms=duration, bandwidth_kbps=0, latency_ms=latency)

    def _determineLatency(self, position1, position2):
        '''
        Calculates distance between two points. Used to calcualte latency.
        '''
        position1 = position1 % 100, position1 // 100
        position2 = position2 % 100, position2 // 100
        distance = round(math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2), 2) # Euklidean distance
        distance = round(distance * 5, 2)
        return distance

    def saveData(self, time, finalStep=False):
        if time == 0: return
        client_ids = [client.id for client in self.clients]
        self.buffered_data.append([self.util.episodeCounter, time, self.id, len(client_ids), np.array(client_ids), self.bandwidth_kbps, \
                                   self.currentBandwidth, self.price, self.money, self.contigent])
        if finalStep:
            self.util.cdnCsvExport(self.buffered_data)
            gym.logger.info('SaveData for cdn %s.' % self.id)