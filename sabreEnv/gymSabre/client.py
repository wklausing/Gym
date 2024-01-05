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
        self.status = 'init'
        self.metrics = []

        # For recordings
        # self.buffData = []
        # self.buffDataStep = {}

    def setManifest(self, manifest):
        self.manifest = manifest
        self.idxManifest = 0
        self.edgeServer = self.cdns[self.manifest[self.idxManifest]]
        self.edgeServer.addClient(self)
        self.needsManifest = False

    def provideNetworkCondition(self, duration_ms, bandwidth_kbps, latency_ms):
        '''
        Here network condition will be provided to Sabre.
        '''
        self.sabre.network.add_network_condition(duration_ms=duration_ms, bandwidth_kbps=bandwidth_kbps, latency_ms=latency_ms)

    def removeNetworkCondition(self):
        self.sabre.network.remove_network_condition()

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

        self.time = time

        # Here QoE will be computed with Sabre metrics. Here different kind of QoE can be defined.
        metrics = self._getSabreMetrics()
        self.status = metrics['status']
        
        if metrics['status'] == 'completed' or metrics['status'] == 'downloadedSegment':
            qoe = self._qoe(metrics)
            metrics['qoe'] = qoe
            metrics['qoeFlag'] = True
        else:
            metrics['qoe'] = 0
            metrics['qoeFlag'] = False

        metrics['episode'] = self.util.episodeCounter
        metrics['gymTime'] = self.time
        metrics['id'] = self.id
        metrics['alive'] = self.alive
        metrics['location'] = self.location
        metrics['manifest'] = self.manifest
        metrics['edgeServer_id'] = self.edgeServer.id
        metrics['edgeServer_location'] = self.edgeServer.location
        metrics['latency'] = self.currentLatency
        metrics['bandwidth'] = self.currentBandwidth

        self.metrics.append(metrics)
        return metrics    
    
    def _qoe(self, metrics):
        '''
        Here QoE will be computed with Sabre metrics. Here different kind of QoE can be defined.
        '''
        if self.qoeMetric == 'bitrate' or True:
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

    def saveData(self, finalStep=False):
        if self.time == -1:
            return
        elif finalStep:
            print('SaveData for client %s.' % self.id)
            self.util.clientCsvExport(self.metrics)
            print('SaveData done for client %s.' % self.id)
        else:
            gym.logger.error('Error in saveData() for clients.')
