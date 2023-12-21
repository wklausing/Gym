import gymnasium as gym
import numpy as np

import os
import csv

class EdgeServer:

    def __init__(self, id, location, filename, episodeCounter, price=1, bandwidth_kbps=1000):
        self.id = id
        self.location = location
        self.price = price
        self.contigent = 0
        self.bandwidth_kbps = bandwidth_kbps
        self.currentBandwidth = bandwidth_kbps
        self.clients = []
        self.money = 0

        self.filename = filename
        self.episodeCounter = episodeCounter
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
    
    def saveData(self, time, finalStep=False):
        if time == 0: return

        client_ids = [client.id for client in self.clients]
        self.buffered_data.append([self.episodeCounter, time, self.id, np.array(client_ids), self.bandwidth_kbps, self.currentBandwidth, self.contigent, self.money])
        
        if finalStep:
            file_exists = os.path.isfile(self.filename) and os.path.getsize(self.filename) > 0
            with open(self.filename, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['episode', 'time', 'id', 'clients', 'bandwidth', 'currentBandwidth', 'contigent', 'money'])
                
                for row in self.buffered_data:
                    writer.writerow(row)