import unittest

from sabreEnv import Sabre

class TestMainFunction(unittest.TestCase):

    def testSabre(self):
        '''
        Testing Sabre with a controll dictionary.
        '''
        sabre = Sabre(abr='throughput', moving_average='ewma', replace='right', abr_osc=False)
        result = sabre.testing()
        controllDict = {'done': True, 'buffer_size': 25000, 'total_played_utility': 437.36238530806844, 'time_average_played_utility': 2.1968726071655786, 'total_played_bitrate': 438411, 'time_average_played_bitrate': 2202.139801989736, 'total_play_time': 597.2522720000001, 'total_play_time_chunks': 199.0840906666667, 'total_rebuffer': 0.0, 'rebuffer_ratio': 0.0, 'time_average_rebuffer': 0.0, 'total_rebuffer_events': 0, 'time_average_rebuffer_events': 0.0, 'total_bitrate_change': 20907, 'time_average_bitrate_change': 105.01592533079555, 'total_log_bitrate_change': 12.7743049135312, 'time_average_log_bitrate_change': 0.06416537288717689, 'time_average_score': 2.1968726071655786, 'total_reaction_time': 194.27003466666656, 'estimate': -284.4087415001153}
        self.assertDictEqual(result, controllDict)
                         

if __name__ == '__main__':
    unittest.main()
