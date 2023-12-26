from sabreEnv.sabre.sabreV8 import Sabre
import math
import gymnasium as gym

import os
import csv

class Client():
    '''
    Manifest is list with the ids of edge-server.
    cdns is list of edge-servers.
    '''

    def __init__(self, id, location, cdns, filename, episodeCounter):
        self.id = id
        self.alive = True
        self.location = location
        self.time = 0
        self.cdns = cdns

        # Sabre implementation
        self.sabre = Sabre(verbose=False)
        self.qoeStatus = 'init'   
        self.qoe = []
        self.weightBitrate = 1
        self.weightBitrateChange = 1
        self.weightRebuffer = 1

        # For recordings
        self.buffered_data = []
        self.filename = filename
        self.episodeCounter = episodeCounter

    def setManifest(self, manifest):
        self.manifest = manifest
        self.idxManifest = 0
        self.edgeServer = self.cdns[self.manifest[self.idxManifest]]
        self.edgeServer.addClient(self)

    def fetchContent(self, time):
        '''
        If fetch origin can delivier than return 1. If not than increase idxCDN to select next server and return -1.
        Here Sabre should be used to get a real reward based on QoE.
        '''
        self.time = time

        # Distance between client and edge-server. Used for latency calculation.
        latency = self._determineLatency(self.location, self.edgeServer.location)

        # Bandwidth
        bandwidth = self.edgeServer.currentBandwidth

        if self.edgeServer.contigent > 0:
            # Add network trace to client
            self.sabre.network.add_network_condition(duration_ms=1000, bandwidth_kbps=bandwidth, latency_ms=latency)
        else:
            gym.logger.info('Not enough contingent for client %s at CDN %s.' % (self.id, self.edgeServer.id))
            self._changeCDN()
            self.sabre.network.remove_network_condition()

        qoe = self._getQoE()
        if qoe['status'] == 'fetching':
            self.edgeServer.deductContigent(qoe['size'])
            self.qoe.append(qoe['qoe'])

        return qoe
        
    def _getQoE(self):
        '''
        Here QoE will be computed with Sabre.
        '''
        print('Entering Sabre.')
        result = self.sabre.downloadSegment()
        if result['status'] == 'downloadedSegment':
            self.qoeStatus = result['status']
            qoe = result['time_average_played_bitrate'] * self.weightBitrate - result['time_average_bitrate_change'] * self.weightBitrateChange - result['time_average_rebuffer_events'] * self.weightRebuffer
            return {'qoe': qoe, 'size': result['size'], 'status': 'fetching'}
        elif result['status'] == 'completed':
            self.qoeStatus = result['status']
            self._setDone()
            return {'status': 'completed'}
        else:
            gym.logger.info('Not enough trace for client %s to fetch from CDN %s.' % (self.id, self.edgeServer.id))
            self.qoeStatus = result['status']
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

    def saveData(self, finalStep=False):
        if self.time == 0: return
        qoe = self.qoe[-1] if len(self.qoe) > 0 and self.qoeStatus != 'missingTrace' else None
        self.buffered_data.append([self.episodeCounter, self.time, self.manifest, self.location, self.edgeServer.id, self.edgeServer.location, self.alive, self.id, self.qoeStatus, qoe])
        if finalStep:
            file_exists = os.path.isfile(self.filename) and os.path.getsize(self.filename) > 0
            with open(self.filename, 'a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['episode', 'time', 'manifest', 'location', 'edgeServer_id', 'edgeServer_location', 'alive', 'id', 'qoeStatus', 'qoe'])
                
                for row in self.buffered_data:
                    writer.writerow(row)
        
        


