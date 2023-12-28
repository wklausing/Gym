from sabreEnv.sabre.sabreV9 import Sabre
import math
import gymnasium as gym

import os
import csv

class Client():
    '''
    Manifest is list with the ids of edge-server.
    cdns is list of edge-servers.
    '''

    def __init__(self, id, location, cdns, util, contentSteering=False, ttl=500):
        self.util = util
        self.id = id
        self.alive = True
        self.location = location
        self.time = -1
        self.cdns = cdns
        self.needsManifest = True
        self.contentSteering = contentSteering
        self.ttl = ttl
        self.qoeMetric = 'bitrate'

        self.currentBandwidth = 0
        self.currentLatency = 0

        # Sabre implementation
        self.sabre = Sabre(verbose=False)
        self.qoeStatus = 'init'   
        self.sabreMetrics = []
        self.delay = 0
        self.weightBitrate = 1
        self.weightBitrateChange = 1
        self.weightRebuffer = 1

        # For recordings
        self.buffData = []
        self.buffDataStep = {}

    def setManifest(self, manifest):
        self.manifest = manifest
        self.idxManifest = 0
        self.edgeServer = self.cdns[self.manifest[self.idxManifest]]
        self.edgeServer.addClient(self)
        self.needsManifest = False

    def step(self, time):
        '''
        Client will do its move. It will download a segment from the edge-server, or do nothing if there is a delay caused by Sabre.
        '''
        # Content steering: Deducting time from TTL 
        if self.time != time and self.contentSteering:
            self.ttl -= 1
            if self.ttl == 0:
                gym.logger.info('Client %s aks for new manifest.' % self.id)
                self.needsManifest = True

        if self.time == time: # Client already did its move
            gym.logger.error('Client %s has delay at time %s.' % (self.id, time))
            result = {'status': 'delay', 'delay': self.delay}
        elif self.delay > 0: # Delay is set by Sabre
            self.delay -= 1
            result = {'status': 'delay', 'delay': self.delay}
        else: # Delay is over
            self.time = time

            # Network conditions
            # Distance between client and edge-server. Used for latency calculation.
            self.currentLatency = self._determineLatency(self.location, self.edgeServer.location)
            self.currentBandwidth = self.edgeServer.currentBandwidth
            self.buffDataStep['network'] = {'latency': self.currentLatency, 'bandwidth': self.currentBandwidth}

            if self.edgeServer.contigent > 0:
                # Add network trace to client
                self.sabre.network.add_network_condition(duration_ms=1000, bandwidth_kbps=self.currentBandwidth, latency_ms=self.currentLatency)
            else:
                gym.logger.info('Not enough contingent for client %s at CDN %s.' % (self.id, self.edgeServer.id))
                self._changeCDN()
                self.sabre.network.remove_network_condition()
            result = self._fetch()

        # Save Data for later
        self.buffData.append(self.buffDataStep)
        self.buffDataStep = {} # Reset

        return result
        
    def _fetch(self):
        '''
        Here QoE will be computed with Sabre metrics. Here different kind of QoE can be defined.
        '''
        metrics = self._getSabreMetrics()
        if metrics['status'] == 'completed' and self.id==0:
            pass

        if metrics['status'] == 'completed' or metrics['status'] == 'downloadedSegment':
            qoe = self._qoe(metrics)
            result = {'status': metrics['status'], 'qoe': qoe}
        elif metrics['status'] == 'delay':
            metrics['delay'] = round(metrics['delay'] / 1000, 0)
            result = {'status': 'delay', 'delay': metrics['delay']}
        elif metrics['status'] == 'missingTrace':
            gym.logger.info('Not enough trace for client %s to fetch from CDN %s.' % (self.id, self.edgeServer.id))
            result = {'status': 'missingTrace'}
        else:
            gym.logger.error('Unknown Sabre status: %s' % metrics['status'])
            quit()

        self.buffDataStep = {**metrics, **result}
        self.buffDataStep['gymTime'] = self.time
        self.buffDataStep['alive'] = self.alive
        self.buffDataStep['episode'] = self.util.episodeCounter
        self.buffDataStep['manifest'] = self.manifest
        self.buffDataStep['location'] = self.location
        self.buffDataStep['edgeServer_id'] = self.edgeServer.id
        self.buffDataStep['edgeServer_location'] = self.edgeServer.location
        self.buffDataStep['id'] = self.id
        self.buffDataStep['latency'] = self.currentLatency
        self.buffDataStep['bandwidth'] = self.currentBandwidth

        return result
    
    def _qoe(self, metrics):
        '''
        Here QoE will be computed with Sabre metrics. Here different kind of QoE can be defined.
        '''
        if self.qoeMetric == 'bitrate':
            qoe = metrics['time_average_played_bitrate'] * self.weightBitrate - metrics['time_average_bitrate_change'] * \
                self.weightBitrateChange - metrics['time_average_rebuffer_events'] * self.weightRebuffer
        return qoe
    
    def _getSabreMetrics(self):
        '''
        Entering Sabre to get calculate metrics.
        '''
        #print('Entering Sabre.')
        result = self.sabre.downloadSegment()
        self.qoeStatus = result['status']
        if result['status'] == 'completed': self._setDone()
        return result
        
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
        gym.logger.info('Client %s downloaded content successfully at time %s.' % (self.id, self.time))
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
        if self.time == -1:
            return
        elif finalStep:
            print('SaveData for client %s.' % self.id)
            self.util.clientCsvExport(self.buffData)
            print('SaveData done for client %s.' % self.id)
        else:
            gym.logger.error('Error in saveData() for clients.')
