import unittest

from sabreEnv.sabre.sabreV1 import init as initSabreV1 # Having this for testing purposes
from sabreEnv.sabre.sabreV2 import init as initSabreV2 # Encapsulation done
from sabreEnv.sabre.sabreV3 import Sabre as SabreV3
from sabreEnv.sabre.sabreV4 import Sabre as SabreV4
from sabreEnv.sabre.sabreV6 import Sabre as SabreV6
from sabreEnv.sabre.sabreV8 import Sabre as SabreV8
from sabreEnv.sabre.sabreV9 import Sabre as SabreV9


class TestMainFunction(unittest.TestCase):

    # Values from original Sabre in default mode.
    originalResult = {
            'buffer_size': 25000,
            'total_played_utility': 484.9654794493675,
            'time_average_played_utility': 2.4359830955119453,
            'total_played_bitrate': 579714,
            'time_average_played_bitrate': 2911.905205778773,
            'total_play_time': 597.252272,
            'total_play_time_chunks': 199.08409066666667,
            'total_rebuffer': 0.0,
            'rebuffer_ratio': 0.0,
            'time_average_rebuffer': 0.0,
            'total_rebuffer_events': 0,
            'time_average_rebuffer_events': 0.0,
            'total_bitrate_change': 80654,
            'time_average_bitrate_change': 405.1252901721904,
            'total_log_bitrate_change': 30.65922013412642,
            'time_average_log_bitrate_change': 0.15400135707207366,
            'time_average_score': 2.4359830955119453,
            'total_reaction_time': 61.26135999999942,
            'estimate': -234.28114307059911
        }

    # def testSabreV1_regression(self):
    #     '''
    #     Testing original values against sabre.py
    #     '''
    #     resultSabreOriginal = initSabreV1()
    #     for key in self.originalResult:
    #         self.assertEqual(self.originalResult[key], resultSabreOriginal[key])

    # def testSabreV2_regression(self):
    #     '''
    #     Testing sabreV1.py against sabreV2.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV1 = initSabreV1(abr=abr, moving_average=average, verboseInput=False)
    #             resultSabreV2 = initSabreV2(abr=abr, moving_average=average, verbose=False)
    #             for key in resultSabreV1:
    #                 self.assertEqual(resultSabreV1[key], resultSabreV2[key])

    # def testSabreV3_regression(self):
    #     '''
    #     Testing sabreV2.py against sabreV3.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV2 = initSabreV2(abr=abr, moving_average=average, verbose=False)
    #             resultSabreV3 = SabreV3(abr=abr, moving_average=average, verbose=False).testing()
    #             for key in resultSabreV2:
    #                 self.assertEqual(resultSabreV2[key], resultSabreV3[key])

    # def testSabreV4_regression(self):
    #     '''
    #     Testing sabreV3.py against sabreV4.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV3 = SabreV3(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             resultSabreV4 = SabreV4(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             for key in resultSabreV3:
    #                 # print(key)
    #                 self.assertEqual(resultSabreV3[key], resultSabreV4[key])
        
    # def testSabreV6_regression(self):
    #     '''
    #     Testing sabreV4.py against sabreV6.py
    #     '''
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV4 = SabreV4(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             resultSabreV6 = SabreV6(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             for key in resultSabreV4:
    #                 # print(key)
    #                 self.assertEqual(resultSabreV4[key], resultSabreV6[key])
                        
    # def testSabreV8_regression(self):
    #     '''
    #     Testing sabreV6.py against sabreV8.py
    #     '''        
    #     abrList = ['bola', 'bolae', 'throughput', 'dynamic', 'dynamicdash']
    #     averageList = ['ewma', 'sliding']
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV6 = SabreV6(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             resultSabreV8 = SabreV8(abr=abr, moving_average=average, verbose=False,  replace='right').testing()
    #             for key in resultSabreV6:
    #                 self.assertEqual(resultSabreV6[key], resultSabreV8[key])

    def testSabreV9_bandwidthResponse(self):
        '''
        Testing sabreV9 and how it responds to bandwidth changes.
        '''        
        resultSabreV9 = SabreV9().testing()
        assert resultSabreV9['total_play_time'] > 0
                
                

if __name__ == '__main__':
    unittest.main()
