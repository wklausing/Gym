import csv
import os
import json
import pandas as pd
import math

class Util:

    def __init__(self, saveData=False, savingPath='sabreEnv/gymSabre/data/', filePrefix='', gridWidth=100, gridHeight=100):
        
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight

        self.clientFilename = savingPath + filePrefix + 'client.csv'
        self.cdnFilename = savingPath + filePrefix + 'cdn.csv'
        self.cpFilename = savingPath + filePrefix + 'cp.csv'

        self.saveData = saveData
        
        # Shared variables
        self.episodeCounter = 0

    def calcDistance(self, position1, position2):
        pos1 = position1 % self.gridWidth, position1 // self.gridHeight
        pos2 = position2 % self.gridWidth, position2 // self.gridHeight
        distance = round(math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2), 2) # Euklidean distance
        return distance

    def cdnCsvExport(self, stepData):
        allCdnKeys = ['episode', 'time', 'id', 'clientsCount', 'clients', 'bandwidth_kbps', 'currentBandwidth', 'price', 'money', 'contigent']
        file_exists = os.path.exists(self.cdnFilename)
        with open(self.cdnFilename, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(allCdnKeys)
            writer.writerows(stepData)

    def clientCsvExport(self, stepData):
        clientData = pd.DataFrame(stepData)
        # Check if the file exists
        if os.path.exists(self.clientFilename):
            # Append without writing the header
            clientData.to_csv(self.clientFilename, mode='a', header=False, index=False)
        else:
            # Write a new file with the header
            clientData.to_csv(self.clientFilename, mode='w', header=True, index=False)


    def load_json(self, path):
        with open(path) as file:
            obj = json.load(file)
        return obj