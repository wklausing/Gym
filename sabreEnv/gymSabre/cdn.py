import gymnasium as gym
import numpy as np

import os
import csv

import math

class EdgeServer:

    def __init__(self, util, id, location, filename, price=1, bandwidth_kbps=1000):
        self.util = util

        self.id = id
        self.location = location
        self.price = price
        self.contigent = 0
        self.bandwidth_kbps = bandwidth_kbps
        self.currentBandwidth = bandwidth_kbps
        self.clients = []
        self.money = 0

        self.filename = filename
        self.buffered_data = []

    def sellContigent(self, cpMoney, amount):
        price = round(self.price * amount,2)
        if cpMoney >= price:
            cpMoney -= price
            self.money += price
            self.contigent += amount * 1000
        return round(cpMoney, 2)
    
    def addClient(self, client):
        self.clients.append(client)
        self.currentBandwidth = round(self.bandwidth_kbps / len(self.clients),2)

    def removeClient(self, client):
        self.clients.remove(client)
        if len(self.clients) == 0: return
        self.currentBandwidth = round(self.bandwidth_kbps / len(self.clients),2)

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
    
    clientsStatus = {}
    def manageClients(self, time):
        '''
        Here clients will fetch content from CDN and return the result as a dictionary.
        '''

        # Check what is needed for clients to proceed
        for client in self.clients:
            if client.status == 'missingTrace':
                # Client needs trace
                size = client.metrics[-1]['size']
            elif client.status == 'downloadedSegment':
                pass
            elif client.status == 'completed':
                self.clientsRequiresTrace =- 1
            elif client.status == 'delay':
                self.clientsRequiresTrace =- 1

        # Distribute bandwidth
        clients = [client for client in self.clients if client.status == 'missingTrace']
        for client in clients:
            duration = self._calcDuration(time, client)
            bandwidth = self.bandwidth_kbps / len(clients)
            latency = self._determineLatency(self.location, client.location)
            client.provideNetworkCondition(duration_ms=duration, bandwidth_kbps=bandwidth, latency_ms=latency)    

        # Let client do its move
        for client in self.clients:
            result = client.step(time)
            self.clientsStatus[client.id] = result


    def _calcDuration(self, time, client):
        return 1000

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
        self.buffered_data.append([self.util.episodeCounter, time, self.id, np.array(client_ids), self.bandwidth_kbps, self.currentBandwidth, self.contigent, self.money])
        
        if finalStep:
            file_exists = os.path.isfile(self.filename) and os.path.getsize(self.filename) > 0
            with open(self.filename, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['episode', 'time', 'id', 'clients', 'bandwidth', 'currentBandwidth', 'contigent', 'money'])
                
                for row in self.buffered_data:
                    writer.writerow(row)