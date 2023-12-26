import unittest

#from sabreEnv.sabre.sabreV6 import Sabre as SabreV6
from sabreEnv.sabre.sabreV8 import Sabre as SabreV8
from sabreEnv.sabre.sabreV9 import Sabre as SabreV9

class TestMainFunction(unittest.TestCase):

    # def testSabreV8Network(self):
    #     '''
    #     Testing sabreV6.py against sabreV8.py, but with different network traces which should have the exact same results.
    #     '''        
    #     abrList = ['bolae']# 'bola', 'throughput', 'dynamic', 'dynamicdash' 
    #     averageList = ['ewma']# ', 'sliding'
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             resultSabreV8 = SabreV8(abr=abr, moving_average=average, verbose=False,  replace='right').testing(network='sabreEnv/sabre/data/networkTest1.json')
    #             resultSabreV8Sec = SabreV8(abr=abr, moving_average=average, verbose=False,  replace='right').testing(network='sabreEnv/sabre/data/networkTest2.json')
    #             for key in resultSabreV8:
    #                 print('key: ', key)
    #                 self.assertEqual(resultSabreV8[key], resultSabreV8Sec[key])
    #                 #self.assertAlmostEqual(resultSabreV8[key], resultSabreV8Sec[key],1)

    def testSabreV9_regression(self):
        abrList = ['bolae','bola', 'throughput', 'dynamic', 'dynamicdash' ]
        averageList = ['ewma', 'sliding']
        for abr in abrList:
            for average in averageList:
                print('Testing: ', abr, average)
                resultSabreV8 = SabreV8(abr=abr, moving_average=average, verbose=False,  replace='right').testing(network='sabreEnv/sabre/data/networkTest1.json')
                resultSabreV9 = SabreV9(abr=abr, moving_average=average, verbose=False,  replace='right').testing(network='sabreEnv/sabre/data/networkTest2.json')
                for key in resultSabreV8:
                    print('key: ', key)
                    if key == 'estimate': continue
                    self.assertEqual(resultSabreV8[key], resultSabreV9[key])

    # def testSabreV9_stepRegression(self):
    #     '''
    #     Testing segment by segment.
    #     '''
    #     abrList = ['bolae']# 'bola', 'throughput', 'dynamic', 'dynamicdash' 
    #     averageList = ['ewma']# ', 'sliding'
    #     for abr in abrList:
    #         for average in averageList:
    #             print('Testing: ', abr, average)
    #             sabreV8 = SabreV8(abr=abr, moving_average=average, verbose=False, replace='right')
    #             sabreV9 = SabreV9(abr=abr, moving_average=average, verbose=False, replace='right')
    #             segment = 0
    #             while True: 
    #                 print('segment: ', segment)
    #                 if segment == 72:
    #                     pass
    #                 resultSabreV8 = sabreV8.testingSegment(network='sabreEnv/sabre/data/networkTest1.json')
    #                 resultSabreV9 = sabreV9.testingSegment(network='sabreEnv/sabre/data/networkTest2.json')
    #                 for key in resultSabreV8:
    #                     print('key: ', key, resultSabreV8[key])
    #                     self.assertEqual(resultSabreV8[key], resultSabreV9[key])
    #                 if resultSabreV8['status'] == 'completed': break
    #                 segment += 1         

if __name__ == '__main__':
    unittest.main()