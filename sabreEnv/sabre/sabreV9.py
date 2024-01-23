from sabreEnv.sabre.sabreV6 import BolaEnh, Ewma, Util, Replace, SessionInfo, SlidingWindow, Bola, ThroughputRule, Dynamic, DynamicDash, Bba
from collections import namedtuple
import math
import numpy as np


class NetworkModel:

    DownloadProgress = namedtuple('DownloadProgress',
                                  'index quality '
                                  'size downloaded '
                                  'time time_to_first_bit '
                                  'abandon_to_quality')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')

    min_progress_size = 12000
    min_progress_time = 50

    permanent = False # If false does also mean that the network trace is empty
    network_total_timeTEMP = 0 #self.util.network_total_time

    def __init__(self, util):
        self.util = util

        self.util.sustainable_quality = None
        self.util.network_total_time = 0
        self.traces = []
        self.networkIndex = -1
        self.networkIndexTEMP = -1
        self.time_to_next = 0
        self.time_to_nextTEMP = 0

    def add_network_condition(self, duration_ms, bandwidth_kbps, latency_ms):
        '''
        Adds a new network condition to self.trace. Will be removed after one use.
        '''
        network_trace = self.NetworkPeriod(time=duration_ms,
                                    bandwidth=bandwidth_kbps *
                                    1,
                                    latency=latency_ms)
        self.traces.append(network_trace)
        self.permanent = True

    def remove_network_condition(self):
        '''
        Adds a new network condition to self.trace. Will be removed after one use.
        '''
        self.traces = []
        self.networkIndex = 0
        self.permanent = False

    def _next_network_period(self):
        '''
        Changes network conditions, according to self.trace
        '''
        self.networkIndexTEMP += 1
        if self.networkIndexTEMP >= len(self.traces):
            self.permanent = False
            self.networkIndexTEMP -= 1
            return False
        self.time_to_nextTEMP = self.traces[self.networkIndexTEMP].time

        latency_factor = 1 - \
            self.traces[self.networkIndexTEMP].latency / self.util.manifest.segment_time
        effective_bandwidth = self.traces[self.networkIndexTEMP].bandwidth * latency_factor

        previous_sustainable_quality = self.util.sustainable_quality
        self.util.sustainable_quality = 0
        for i in range(1, len(self.util.manifest.bitrates)):
            if self.util.manifest.bitrates[i] > effective_bandwidth:
                break
            self.util.sustainable_quality = i
        if (self.util.sustainable_quality != previous_sustainable_quality and
                previous_sustainable_quality != None):
            self.util.advertize_new_network_quality(
                self.util.sustainable_quality, previous_sustainable_quality, self.network_total_timeTEMP)
        return True

    def _do_latency_delay(self, delay_units):
        '''
        Return delay time.
        This needs to return false if new network trace is required.
        '''
        total_delay = 0
        while delay_units > 0:
            current_latency = self.traces[self.networkIndexTEMP].latency # 200
            time = delay_units * current_latency
            if time <= self.time_to_nextTEMP:
                total_delay += time
                self.network_total_timeTEMP += time
                self.time_to_nextTEMP -= time
                delay_units = 0
            else:
                total_delay += self.time_to_nextTEMP
                self.network_total_timeTEMP += self.time_to_nextTEMP
                delay_units -= self.time_to_nextTEMP / current_latency
                self._next_network_period()
                if self.permanent == False: return 0
        return total_delay

    def _do_download(self, size):
        '''
        Return download time
        '''
        total_download_time = 0
        while size >= 0:
            current_bandwidth = self.traces[self.networkIndexTEMP].bandwidth
            if size <= self.time_to_nextTEMP * current_bandwidth:
                time = size / current_bandwidth
                total_download_time += time
                self.network_total_timeTEMP += time
                self.time_to_nextTEMP -= time
                break
            else:
                total_download_time += self.time_to_nextTEMP
                self.network_total_timeTEMP += self.time_to_nextTEMP
                size -= self.time_to_nextTEMP * current_bandwidth
                self._next_network_period()
                if self.permanent == False: break
        return total_download_time
        
    def testDownload(self, size, trace):
        '''
        Test if download is possible for given size and trace.
        '''
        index = self.networkIndexTEMP
        time_to_next = self.time_to_nextTEMP
        trace = self.traces

        if trace == []: 
            return False

        ### Latency
        delay_units = 1
        total_delay = 0
        while delay_units > 0:
            current_latency = trace[index].latency
            time = delay_units * current_latency
            if time <= time_to_next:
                total_delay += time
                time_to_next -= time
                delay_units = 0
            else:
                total_delay += time_to_next
                delay_units -= time_to_next / current_latency
                index += 1
                if index >= len(trace): 
                    return False
                time_to_next = trace[index].time
                
        ### Download
        total_trace_time = sum([t.time for t in trace])

        if total_trace_time > total_delay:
            index = 0
            
            total_download_time = 0
            while size >= 0:
                current_bandwidth = trace[index].bandwidth
                if size <= time_to_next * current_bandwidth:
                    time = size / current_bandwidth
                    total_download_time += time
                    time_to_next -= time
                    break
                else:
                    total_download_time += time_to_next
                    size -= time_to_next * current_bandwidth
                    index += 1
                    if index >= len(trace): return False
                    time_to_next = trace[index].time
            return total_download_time + total_delay
        return False

    def _do_minimal_latency_delay(self, delay_units, min_time):
        total_delay_units = 0
        total_delay_time = 0
        while delay_units > 0 and min_time > 0:
            current_latency = self.traces[self.networkIndexTEMP].latency
            time = delay_units * current_latency
            if time <= min_time and time <= self.time_to_nextTEMP:
                units = delay_units
                self.time_to_nextTEMP -= time
                self.network_total_timeTEMP += time
            elif min_time <= self.time_to_nextTEMP:
                # time > 0 implies current_latency > 0
                time = min_time
                units = time / current_latency
                self.time_to_nextTEMP -= time
                self.network_total_timeTEMP += time
            else:
                time = self.time_to_nextTEMP
                units = time / current_latency
                self.network_total_timeTEMP += time
                self._next_network_period()
                if self.permanent == False: break
            total_delay_units += units
            total_delay_time += time
            delay_units -= units
            min_time -= time

        return (total_delay_units, total_delay_time)

    def _do_minimal_download(self, size, min_size, min_time):
        total_size = 0
        total_time = 0
        while size > 0 and (min_size > 0 or min_time > 0):
            current_bandwidth = self.traces[self.networkIndexTEMP].bandwidth
            if current_bandwidth > 0:
                min_bits = max(min_size, min_time * current_bandwidth)
                bits_to_next = self.time_to_nextTEMP * current_bandwidth
                if size <= min_bits and size <= bits_to_next:
                    bits = size
                    time = bits / current_bandwidth
                    self.time_to_nextTEMP -= time
                    self.network_total_timeTEMP += time
                elif min_bits <= bits_to_next:
                    bits = min_bits
                    time = bits / current_bandwidth
                    # make sure rounding error does not push while loop into endless loop
                    min_size = 0
                    min_time = 0
                    self.time_to_nextTEMP -= time
                    self.network_total_timeTEMP += time
                else:
                    bits = bits_to_next
                    time = self.time_to_nextTEMP
                    self.network_total_timeTEMP += time
                    self._next_network_period()
            else:  # current_bandwidth == 0
                bits = 0
                if min_size > 0 or min_time > self.time_to_next:
                    time = self.time_to_nextTEMP
                    self.network_total_timeTEMP += time
                    self._next_network_period()
                else:
                    time = min_time
                    self.time_to_nextTEMP -= time
                    self.network_total_timeTEMP += time
            
            if self.permanent == False: break

            total_size += bits
            total_time += time
            size -= bits
            min_size -= bits
            min_time -= time
        return (total_size, total_time)

    def delayNet(self, time):
        '''
        I think, it is the delay till the next download. 

        self.util.network_total_time --> network_total_time
        self.time_to_next --> time_to_next
        '''

        if self.permanent == False: return False

        while time > self.time_to_nextTEMP:
            time -= self.time_to_nextTEMP
            self.network_total_timeTEMP += self.time_to_nextTEMP
            self._next_network_period()
            if self.permanent == False: break
        self.time_to_nextTEMP -= time
        self.network_total_timeTEMP += time

        #TODO: Check if this is correct
        self.util.network_total_time = self.network_total_timeTEMP
        self.time_to_next = self.time_to_nextTEMP

    def downloadNet(self, size, idx, quality, buffer_level, check_abandon=None):
        '''
        Returns tuple of DownloadProgress.
        '''
        if self.permanent == False: return False

        downloadProgress = False # Assuming not enough trace 
        self.network_total_timeTEMP = self.util.network_total_time
        self.time_to_nextTEMP = self.time_to_next
        self.networkIndexTEMP = self.networkIndex

        if size <= 0:# If size is not positive, than return 
            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                         size=0, downloaded=0,
                                         time=0, time_to_first_bit=0,
                                         abandon_to_quality=None)
        elif not check_abandon or (NetworkModel.min_progress_time <= 0 and
                                 NetworkModel.min_progress_size <= 0):
            
            latency = self._do_latency_delay(1)
            if self.permanent == False: return False

            time = latency + self._do_download(size)
            if self.permanent == False: return False
            
            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                         size=size, downloaded=size,
                                         time=time, time_to_first_bit=latency,
                                         abandon_to_quality=None)
        else:
            total_download_time = 0
            total_download_size = 0
            min_time_to_progress = NetworkModel.min_progress_time
            min_size_to_progress = NetworkModel.min_progress_size

            if NetworkModel.min_progress_size > 0:
                latency = self._do_latency_delay(1)
                total_download_time += latency
                min_time_to_progress -= total_download_time
                delay_units = 0
            else:
                latency = None
                delay_units = 1

            abandon_quality = None
            while total_download_size < size and abandon_quality == None:
                if self.permanent == False: break

                if delay_units > 0:
                    # NetworkModel.min_progress_size <= 0
                    (units, time) = self._do_minimal_latency_delay(
                        delay_units, min_time_to_progress)
                    total_download_time += time
                    delay_units -= units
                    min_time_to_progress -= time
                    if delay_units <= 0:
                        latency = total_download_time

                if delay_units <= 0:
                    # don't use else to allow fall through
                    (bits, time) = self._do_minimal_download(size - total_download_size,
                                                            min_size_to_progress, min_time_to_progress)
                    total_download_time += time
                    total_download_size += bits
                    # no need to upldate min_[time|size]_to_progress - reset below

                dp = self.DownloadProgress(index=idx, quality=quality,
                                        size=size, downloaded=total_download_size,
                                        time=total_download_time, time_to_first_bit=latency,
                                        abandon_to_quality=None)
                if total_download_size < size:
                    abandon_quality = check_abandon(
                        dp, max(0, buffer_level - total_download_time))
                    if abandon_quality != None:
                        if self.util.verbose:
                            print('%d abandoning %d->%d' %
                                (idx, quality, abandon_quality))
                            print('%d/%d %d(%d)' %
                                (dp.downloaded, dp.size, dp.time, dp.time_to_first_bit))
                    min_time_to_progress = NetworkModel.min_progress_time
                    min_size_to_progress = NetworkModel.min_progress_size

            downloadProgress = self.DownloadProgress(index=idx, quality=quality,
                                        size=size, downloaded=total_download_size,
                                        time=total_download_time, time_to_first_bit=latency,
                                        abandon_to_quality=abandon_quality)
            
        if self.permanent:
            self.util.network_total_time = self.network_total_timeTEMP
            self.time_to_next = self.time_to_nextTEMP
            self.networkIndex = self.networkIndexTEMP
            return downloadProgress
        else:
            return False        


class Sabre():

    average_default = 'ewma'
    average_list = {}
    average_list['ewma'] = Ewma
    average_list['sliding'] = SlidingWindow

    abr_default = 'bolae'
    abr_list = {}
    abr_list['bola'] = Bola
    abr_list['bolae'] = BolaEnh
    abr_list['throughput'] = ThroughputRule
    abr_list['dynamic'] = Dynamic
    abr_list['dynamicdash'] = DynamicDash
    abr_list['bba'] = Bba

    ManifestInfo = namedtuple('ManifestInfo', 'segment_time bitrates utilities segments')
    NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency permanent')

    util = Util()
    throughput_history = None
    abr = None
    firstSegment = True
    next_segment = 0

    def __init__(
        self,
        abr='bolae',
        abr_basic=False,
        abr_osc=False,
        gamma_p=5,
        half_life=[3, 8],
        max_buffer=25,
        movie='sabreEnv/sabre/data/movie.json',
        movie_length=None,
        moving_average=average_default,
        network='sabreEnv/sabre/data/network.json',
        network_multiplier=1,
        no_abandon=False,
        no_insufficient_buffer_rule=False,
        rampup_threshold=None,
        replace='none',
        seek=None,
        verbose=False,
        window_size=[3]
    ):  
        self.no_abandon = no_abandon
        self.seek = seek

        self.util = Util()
        self.util.verbose = verbose
        self.util.buffer_contents = []
        self.util.buffer_fcc = 0
        self.util.pending_quality_up = []
        self.util.rebuffer_event_count = 0
        self.util.rebuffer_time = 0
        self.util.played_utility = 0
        self.util.played_bitrate = 0
        self.util.total_play_time = 0
        self.util.total_bitrate_change = 0
        self.util.total_log_bitrate_change = 0
        self.util.total_reaction_time = 0
        self.util.last_played = None

        self.overestimate_count = 0
        self.overestimate_average = 0
        self.goodestimate_count = 0
        self.goodestimate_average = 0
        self.estimate_average = 0

        self.util.rampup_origin = 0
        self.util.rampup_time = None
        self.util.rampup_threshold = rampup_threshold

        self.util.max_buffer_size = max_buffer * 1000

        self.time_average_played_bitrateList = []

        self.util.manifest = self.util.load_json(movie)
        bitrates = self.util.manifest['bitrates_kbps']
        self.bitrates = bitrates
        utility_offset = 0 - math.log(bitrates[0])  # so utilities[0] = 0
        utilities = [math.log(b) + utility_offset for b in bitrates]
        self.util.manifest = self.ManifestInfo(segment_time=self.util.manifest['segment_duration_ms'],
                                    bitrates=bitrates,
                                    utilities=utilities,
                                    segments=self.util.manifest['segment_sizes_bits'])
        SessionInfo.manifest = self.util.manifest

        self.buffer_size = max_buffer * 1000
        self.gamma_p = gamma_p

        config = {'buffer_size': self.buffer_size,
                'gp': gamma_p,
                'abr_osc': abr_osc,
                'abr_basic': abr_basic,
                'no_ibr': no_insufficient_buffer_rule}
        
        self.abr_list[abr].use_abr_o = abr_osc
        self.abr_list[abr].use_abr_u = not abr_osc
        self.abr = self.abr_list[abr](config, self.util)

        self.network = NetworkModel(self.util)

        self.replacer = Replace(1, self.util)

        config = {'window_size': window_size, 'half_life': half_life}
        self.throughput_history = self.average_list[moving_average](config, self.util)

        self.foo = True

    def downloadSegment(self):

        # Final playout of buffer at the end.
        if self.next_segment == len(self.util.manifest.segments):
            self.util.playout_buffer()
            result = self.createMetrics()
            result['download_time'] = 0
            result['status'] = 'completed'
            return result
            
        # Download first segment
        if self.firstSegment:
            quality = self.abr.get_first_quality()
            size = self.util.manifest.segments[0][quality]
            self.sizeTEMP = size

            download_metric = self.network.downloadNet(size, 0, quality, 0, None)
            if download_metric == False: return {'status': 'missingTrace', 'size': size}

            download_time = download_metric.time - download_metric.time_to_first_bit
            self.util.buffer_contents.append(download_metric.quality)
            t = download_metric.size / download_time # t represents throughput per ms
            l = download_metric.time_to_first_bit
            self.throughput_history.push(download_time, t, l)
            self.util.total_play_time += download_metric.time
            self.firstSegment = False
            self.next_segment = 1
            self.abandoned_to_quality = None
        else:
            # Download rest of segments

            # do we have space for a new segment on the buffer?
            if self.foo:
                full_delay = self.util.get_buffer_level() + self.util.manifest.segment_time - self.buffer_size

                if full_delay > 0:
                    self.network.delayNet(full_delay)
                    self.util.deplete_buffer(full_delay)
                    self.abr.report_delay(full_delay)
                    if self.util.verbose:
                        print('full buffer delay %d bl=%d' %
                            (full_delay, self.util.get_buffer_level()))
                    return {'status': 'delay', 'delay': full_delay}

                if self.abandoned_to_quality == None:
                    (quality, delay) = self.abr.get_quality_delay(self.next_segment)#8, 0
                    replace = self.replacer.check_replace(quality)
                else:
                    (quality, delay) = (self.abandoned_to_quality, 0)
                    replace = None
                    self.abandon_to_quality = None

                if replace != None:
                    delay = 0
                    current_segment = self.next_segment + replace
                    check_abandon = self.replacer.check_abandon
                else:
                    current_segment = self.next_segment
                    check_abandon = self.abr.check_abandon
                if self.no_abandon:
                    check_abandon = None

                size = self.util.manifest.segments[current_segment][quality]

                self.qualityTEMP = quality
                self.replaceTemp = replace
                self.current_segmentTemp = current_segment
                self.check_abandonTEMP = check_abandon
                self.sizeTEMP = size

                if delay > 0:
                    self.network.delayNet(delay)
                    self.util.deplete_buffer(delay)
                    if self.util.verbose:
                        print('abr delay %d bl=%d' % (delay, self.util.get_buffer_level()))
                    return {'status': 'delay', 'delay': delay}
                    
            else:
                quality = self.qualityTEMP
                replace = self.replaceTemp
                current_segment = self.current_segmentTemp
                check_abandon = self.check_abandonTEMP
                size = self.sizeTEMP
                self.foo = True
            
            download_metric = self.network.downloadNet(size, current_segment, quality,
                                            self.util.get_buffer_level(), check_abandon)
            if download_metric == False: 
                self.foo = False
                return {'status': 'missingTrace', 'size': size}

            self.util.deplete_buffer(download_metric.time)

            # Update buffer with new download
            if replace == None:
                if download_metric.abandon_to_quality == None:
                    self.util.buffer_contents += [quality]
                    self.next_segment += 1
                else:
                    self.abandon_to_quality = download_metric.abandon_to_quality
            else:
                if download_metric.abandon_to_quality == None:
                    if self.util.get_buffer_level() + self.util.manifest.segment_time * replace >= 0:
                        self.util.buffer_contents[replace] = quality
                    else:
                        print('WARNING: too late to replace')
                        pass
                else:
                    pass
                # else: do nothing because segment abandonment does not suggest new download

            if self.util.verbose:
                print('->%d' % self.util.get_buffer_level())

            self.abr.report_download(download_metric, replace != None)

            # calculate throughput and latency
            download_time = download_metric.time - download_metric.time_to_first_bit
            t = download_metric.downloaded / download_time
            l = download_metric.time_to_first_bit

            # check accuracy of throughput estimate
            if self.util.throughput > t:
                self.overestimate_count += 1
                self.overestimate_average += (self.util.throughput - t -
                                        self.overestimate_average) / self.overestimate_count
            else:
                self.goodestimate_count += 1
                self.goodestimate_average += (t - self.util.throughput -
                                        self.goodestimate_average) / self.goodestimate_count
            self.estimate_average += ((self.util.throughput - t - self.estimate_average) /
                                (self.overestimate_count + self.goodestimate_count))

            # update throughput estimate
            if download_metric.abandon_to_quality == None:
                self.throughput_history.push(download_time, t, l)

            # loop while next_segment < len(manifest.segments)
        
        result = self.createMetrics()
        result['status'] = 'downloadedSegment'
        result['download_time'] = download_time
        return result
    
    def createMetrics(self):
        to_time_average = 1 / (self.util.total_play_time / self.util.manifest.segment_time)
        results_dict = {
            'size': self.sizeTEMP,
            'buffer_size': self.buffer_size,
            'buffer_level': self.util.get_buffer_level(),
            'total_played_utility': self.util.played_utility,
            'time_average_played_utility': self.util.played_utility * to_time_average,
            'total_played_bitrate': self.util.played_bitrate,
            'time_average_played_bitrate': self.util.played_bitrate * to_time_average,
            'total_play_time': self.util.total_play_time / 1000,
            'total_play_time_chunks': self.util.total_play_time / self.util.manifest.segment_time,
            'total_rebuffer': self.util.rebuffer_time / 1000,
            'rebuffer_ratio': self.util.rebuffer_time / self.util.total_play_time,
            'time_average_rebuffer': self.util.rebuffer_time / 1000 * to_time_average,
            'total_rebuffer_events': self.util.rebuffer_event_count,
            'time_average_rebuffer_events': self.util.rebuffer_event_count * to_time_average,
            'total_bitrate_change': self.util.total_bitrate_change,
            'time_average_bitrate_change': self.util.total_bitrate_change * to_time_average,
            'total_log_bitrate_change': self.util.total_log_bitrate_change,
            'time_average_log_bitrate_change': self.util.total_log_bitrate_change * to_time_average,
            'time_average_score': to_time_average * (self.util.played_utility - self.gamma_p * self.util.rebuffer_time / self.util.manifest.segment_time),
            'total_reaction_time': self.util.total_reaction_time / 1000,
            'estimate': self.estimate_average,
        }
        if self.util.verbose:
            print(results_dict)

        self.time_average_played_bitrateList.append(results_dict['time_average_played_bitrate'])
        quality_std_dev = np.std(self.time_average_played_bitrateList)

        


        # Calculate QoE metric
        if results_dict['total_rebuffer_events'] > 0:
            first_part = (7/8) * max((math.log(results_dict['total_rebuffer_events']) / 6) + 1, 0)
        else:
            first_part = 0 
        second_part = (1/8) * (min(results_dict['time_average_rebuffer_events'], 15) / 15)
        F_ij = first_part + second_part

        qoe = 5.67 * results_dict['time_average_played_bitrate'] / self.util.manifest.bitrates[-1] \
            - 6.72 * quality_std_dev / self.util.manifest.bitrates[-1] + 0.17 - 4.95 * F_ij

        results_dict['qoe'] = qoe

        return results_dict
    
    def testing(self, network='sabreEnv/sabre/data/network.json'):
        '''
        Runs till everything from manifest is downloaded.
        '''
        network_trace = self.util.load_json(network)
        networkLen = len(network_trace)
        i = 0
        while True:
            #print('segment:', self.next_segment)
            result = self.downloadSegment()

            if result['status'] == 'completed':
                return result
            elif result['status'] == 'downloadedSegment' or result['status'] == 'delay':
                pass
            else:
                trace = network_trace[i]
                self.network.add_network_condition(trace['duration_ms'], trace['bandwidth_kbps'] ,trace['latency_ms'])
                i += 1
                if i == networkLen: i = 0   

    def testNetworkTester():
        '''
        Testing if download time from testDownload is correct.
        '''
        NetworkPeriod = namedtuple('NetworkPeriod', 'time bandwidth latency')
        network_trace = NetworkPeriod(time=100,
                                bandwidth=5000,
                                latency=175)
    
        network_trace2 = NetworkPeriod(time=1,
                                    bandwidth=5000,
                                    latency=75)
        trace = []
        trace.append(network_trace2)
        trace.append(network_trace)
        trace.append(network_trace2)
        trace.append(network_trace)
    
        time = sabre.network.testDownload(886360, trace)
        print(time)

if __name__ == '__main__':
    sabre = Sabre(verbose=False, abr='bolae', moving_average='sliding', replace='right')
    sabre.testing(network='sabreEnv/sabre/data/networkTest1.json')
