from sabreEnv.sabre.sabreV9 import Sabre
import gymnasium as gym
from collections import deque

class Client():
    '''
    Manifest is list with the ids of edge-server.
    cdns is list of edge-servers.

    Possible returns from sabre:
    - missingTrace: Sabre does not have enough information to make a decision.
    - downloadedSegment: Sabre has downloaded a segment.
    - completed: Sabre has downloaded all segments.
    - abortedStreaming: Sabre has aborted streaming.
    - delay: Sabre has a delay, because it buffered enough content already.
    '''
    def __init__(self, id, location, cdns, util, contentSteering=False, ttl=60, bufferSize=25, maxActiveClients=10):
        self.util = util
        self.id = id
        self.alive = True
        self.location = location
        self.time = -1
        self.cdns = cdns
        self.cdn = None
        self.needsManifest = True
        self.csActive = contentSteering
        self.ttl = ttl
        self.qoeMeasure = 'bitrate'
        self.manifest = []
        self.missingTraceTime = 0
        self.maxActiveClients = maxActiveClients

        self.network_conditions = deque()
        self.currentBandwidth = 0
        self.currentLatency = 0
        self.average_bandwidth = None

        # Sabre implementation
        self.sabre = Sabre(max_buffer=bufferSize)
        self.bufferSize = bufferSize
        self.status = 'init' 
        self.delay = 0
        self.metrics = []# Includes also step information.

        self._determineNormalizedQoE()

    def setManifest(self, manifest):
        '''
        Remove duplicates from manifest.
        '''
        unique_list = []
        seen = set()
        for item in manifest:
            if item not in seen:
                unique_list.append(item)
                seen.add(item)

        self.manifest = unique_list
        self.idxManifest = 0
        self.cdn = self.cdns[self.manifest[self.idxManifest]]
        self.cdn.addClient(self)
        self.needsManifest = False
        gym.logger.info('Client %s got manifest %s.' % (self.id, self.manifest))

    def provideNetworkCondition(self, duration_ms, bandwidth_kbps, latency_ms):
        '''
        Here network condition will be provided to Sabre.
        '''
        self.currentBandwidth = bandwidth_kbps
        self.currentLatency = latency_ms
        self.updateNetworkAverages(duration_ms, bandwidth_kbps, latency_ms)
        self.sabre.network.add_network_condition(duration_ms=duration_ms, bandwidth_kbps=bandwidth_kbps, latency_ms=latency_ms)

    def updateNetworkAverages(self, duration_ms, bandwidth_kbps, latency_ms):
        '''
        Network conditions is the reason for a client to change CDN. 
        Therefore, the average of the last 10 network conditions is calculated.
        '''
        self.network_conditions.append((duration_ms, bandwidth_kbps, latency_ms))

        # Remove outdated network conditions
        while len(self.network_conditions) > 10:
            self.network_conditions.popleft()

        # Calculate the averages of the remaining conditions
        if self.network_conditions:
            total_bandwidth = sum(bandwidth for _, bandwidth, _ in self.network_conditions)
            total_latency = sum(latency for _, _, latency in self.network_conditions)
            self.average_bandwidth = round(total_bandwidth / len(self.network_conditions),2)
            self.average_latency = round(total_latency / len(self.network_conditions),2)

    def removeNetworkCondition(self):
        '''
        Not used, but here network condition will be removed from Sabre. 
        Could be used when i.e. client changes CDN. 
        '''
        self.sabre.network.remove_network_condition()

    def step(self, time):
        '''
        Client will do its move. It will download a segment from the edge-server, or do nothing if there is a delay caused by Sabre.
        '''
        # Content steering: Deducting time from TTL 
        if self.csActive and self.time != time:
            self.ttl -= 1
            if self.ttl == 0:
                gym.logger.info('Client %s aks for new manifest.' % self.id)
                self.needsManifest = True

        if self.time == time: # Client already did its move
            metrics = {'status': 'delay', 'delay': self.delay}
        elif self.delay > 0: # Delay is set by Sabre
            self.delay -= 1000
            if self.delay < 0: self.delay = 0
            metrics = {'status': 'delay', 'delay': self.delay}
        else: # Delay is over
            self.time = time

            # Here QoE will be computed with Sabre metrics. Here different kind of QoE can be defined.
            metrics = self.sabre.downloadSegment()
            if metrics['status'] == 'completed': self._setDone()

            if metrics['status'] in ['completed', 'downloadedSegment']:
                score = metrics['time_average_score']
                normalized_value = (score - self.minQoE) / (self.maxQoE - self.minQoE)
                metrics['normalized_qoe'] = normalized_value
                if normalized_value > self.maxQoE:
                    pass
                
            self.status = metrics['status']
            
            # Check if client wants to change CDN
            if self.average_bandwidth != None and self.alive: self._evaluateAndSwitchServer()

            # Check if client wants to abort streaming
            if metrics['status'] == 'missingTrace': 
                self.missingTraceTime += 1
                if self.missingTraceTime >= 10: 
                    gym.logger.info('Client %s aborted streaming.' % self.id)
                    self.alive = False
                    self.cdn.removeClient(self)
                    self.status = 'abortedStreaming'
                    metrics['status'] = 'abortedStreaming'
            else:
                self.missingTraceTime = 0

            if metrics['status'] == 'delay':
                self.delay = metrics['delay']

        # Save data for later
        stepInfos = {
            'episode': self.util.episodeCounter,
            'gymTime': self.time,
            'id': self.id,
            'alive': self.alive,
            'location': self.location,
            'manifest': self.manifest,
            'latency': self.currentLatency,
            'bandwidth': self.currentBandwidth
        }
        self.currentBandwidth = 0
        self.currentLatency = 0
        metrics.update(stepInfos)
        if self.cdn is not None:
            metrics['cdn_id'] = self.cdn.id
            metrics['cdn_location'] = self.cdn.location
        
        self.metrics.append(metrics)

        gym.logger.info('Client %s at time %s is currently in state %s.' % (self.id, time, self.status))
        return metrics 
    
    def _qoe(self, metrics):
        '''
        Here QoE will be computed with Sabre metrics. Here different kind of QoE can be defined.
        '''
        if self.qoeMeasure == 'bitrate':
            qoe = metrics['time_average_played_bitrate'] - metrics['time_average_bitrate_change'] * \
                - metrics['time_average_rebuffer_events']
        return qoe
    
    def _evaluateAndSwitchServer(self):
        '''
        Need to evaluate if client wants to switch CDN.
        '''
        if self.average_bandwidth < 200 or self.average_latency > 2000:
            gym.logger.info('Client %s wants to change CDN because of bad network conditions. (%s,%s)' % (self.id, self.average_bandwidth, self.average_latency))
            self._changeCDN()
        
    def _changeCDN(self):
        '''
        Change CDN if i.e. QoE is bad.
        Iterates over the manifest to select next CDN. When reaching the end it stops.
        '''
        previousCDN = self.idxManifest
        self.idxManifest += 1
        if self.idxManifest >= len(self.manifest):
            self.idxManifest -= 1
            gym.logger.info('Client %s is already at last CDN (id=%s) in manifest, so it cannot change CDN anymore.' % (self.id, self.manifest[self.idxManifest]))
        else:
            self.cdn.removeClient(self)
            self.cdn = self.cdns[self.manifest[self.idxManifest]]
            self.cdn.addClient(self)
            gym.logger.info('Client %s changed from CDN %s to %s.' % (self.id, self.manifest[previousCDN], self.manifest[self.idxManifest]))
        
    def _setDone(self):
        gym.logger.info('Client %s completed download successfully at time %s.' % (self.id, self.time))
        self.alive = False
        self.cdn.removeClient(self)

    def _determineQoE(self, bandwidth, latency):
        '''
        Determines QoE with Sabre for given network conditions. Used to normalize QoE.
        '''
        sabre = Sabre(max_buffer=self.bufferSize, movie='sabreEnv/sabre/data/movie_short_10Seg.json')
        #sabre.network.add_network_condition(duration_ms=99999999, bandwidth_kbps=bandwidth, latency_ms=latency)
        return sabre.determineQoE(bandwidth, latency)
    
    def _determineNormalizedQoE(self):
        '''
        Determines the normalized QoE of the clients.
        '''
        latencyList = []
        for cdn in self.cdns:
            latencyList.append(cdn._determineLatency(cdn.location, self.location))
        bandwidth = self.cdns[0].bandwidth_kbps
        sabre = Sabre(max_buffer=self.bufferSize, movie='sabreEnv/sabre/data/movie_short_10Seg.json')
        self.maxQoE = sabre.determineQoE(bandwidth, min(latencyList))
        sabre = Sabre(max_buffer=self.bufferSize, movie='sabreEnv/sabre/data/movie_short_10Seg.json')
        self.minQoE = sabre.determineQoE(bandwidth/self.maxActiveClients, max(latencyList))

    def saveData(self, finalStep=False):
        if self.time == -1:
            pass
        elif finalStep:
            self.util.clientCsvExport(self.metrics)
            gym.logger.info('SaveData for client %s.' % self.id)
        else:
            gym.logger.error('Error in saveData() for clients.')


if __name__ == "__main__":
    client = Client(1, 'test', [1,2,3], None)
    print(client._determineNormalizedQoE())