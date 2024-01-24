from sabreEnv.gymSabre.gymsabre import GymSabreEnv
from stable_baselines3 import PPO, A2C
from gymnasium.wrappers import FlattenObservation

class FooUtils:
    '''
    Serves merely as a note for scenarios and their respective parameters.
    '''

    def scenario1(self):
        '''
        Scenario 1: 4 CDNs with 10 clients. Goals is to maximize reward with and without Content Steering.
        All clients fetch the same content. 
        '''
        envCsOff = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False, saveData=True, savingPath='sabreEnv/utils/data/', filePrefix='sc1_CS_Off_')
        envCsOff = FlattenObservation(envCsOff)
        envCsOn = GymSabreEnv(ttl=5, cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False, saveData=True, savingPath='sabreEnv/utils/data/', filePrefix='sc1_CS_ON_')
        envCsOn = FlattenObservation(envCsOn)
        return envCsOff, envCsOn

    def scenario2(self):
        '''
        Scenario 2: 4 CDNs with 10 clients. Buffer is 5 seconds. Goals is to maximize reward with and without Content Steering.
        '''
        pass
