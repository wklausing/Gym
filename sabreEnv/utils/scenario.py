from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from gymnasium.wrappers import FlattenObservation
from datetime import datetime

class Scenarios:
    '''
    Serves merely as a note for scenarios and their respective parameters.
    '''
    def __init__(self):
        self.current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.env_name = 'gymsabre'

    def runEnvi(self, env, model, max_steps=1000):
        obs = env.reset()
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            observation, reward, terminated, info = env.step(action)
            if terminated:
                observation = env.reset()
        env.close()

    def scenario1(self, max_steps=1000):
        '''
        Scenario 1: 4 CDNs with 10 clients. Goals is to maximize reward with and without Content Steering.
        All clients fetch the same content. 
        '''
        envCsOff = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False)
        envCsOff = FlattenObservation(envCsOff)
        modelCsOff = PPO('MlpPolicy', envCsOff).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc1/ppo_CsOff/'
        modelPath = path + 'envCsOff_' + self.current_date
        modelCsOff.save(modelPath)
        envCsOff = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False, \
                               saveData=True, savingPath=path, filePrefix='sc1_CS_Off_', \
                               cdnLocationsFixed=[3333, 3366, 6633, 6666])
        envCsOff = FlattenObservation(envCsOff)
        model = PPO.load(modelPath, env=envCsOff)        
        envCsOff = model.get_env()
        self.runEnvi(envCsOff, model, max_steps)

        envCsOn = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=10, contentSteering=True)
        envCsOn = FlattenObservation(envCsOn)
        modelCsOn = PPO('MlpPolicy', envCsOn).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc1/ppo_CsOn/'
        modelPath = path + 'envCsOn_' + self.current_date
        modelCsOn.save(modelPath)
        envCsOn = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False, \
                               saveData=True, savingPath=path, filePrefix='sc1_CS_On_', \
                                cdnLocationsFixed=[3333, 3366, 6633, 6666])
        envCsOn = FlattenObservation(envCsOn)
        model = PPO.load(modelPath, env=envCsOn)        
        envCsOn = model.get_env()
        self.runEnvi(envCsOn, model, max_steps)     

    def scenario2(self, max_steps=1000):
        '''
        Scenario 2: 4 CDNs with 10 clients. Buffer is 5 seconds. Goals is to maximize reward with and without Content Steering.
        '''
        envCsOff = GymSabreEnv(bufferSize=5, cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False)
        envCsOff = FlattenObservation(envCsOff)
        modelCsOff = PPO('MlpPolicy', envCsOff).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc2/ppo_CsOff/'
        modelPath = path + 'envCsOff_' + self.current_date
        modelCsOff.save(modelPath)
        envCsOff = GymSabreEnv(bufferSize=5, cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False, \
                               saveData=True, savingPath=path, filePrefix='sc2_CS_Off_', \
                                cdnLocationsFixed=[3333, 3366, 6633, 6666])
        envCsOff = FlattenObservation(envCsOff)
        model = PPO.load(modelPath, env=envCsOff)        
        envCsOff = model.get_env()
        self.runEnvi(envCsOff, model, max_steps) 

        envCsOn = GymSabreEnv(bufferSize=5, cdns=4, maxActiveClients=10, totalClients=10, contentSteering=True)
        envCsOn = FlattenObservation(envCsOn)
        modelCsOn = PPO('MlpPolicy', envCsOn).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc2/ppo_CsOn/'
        modelPath = path + 'envCsOn_' + self.current_date
        modelCsOn.save(modelPath)
        envCsOn = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=10, contentSteering=False, \
                               saveData=True, savingPath=path, filePrefix='sc2_CS_On_', \
                                cdnLocationsFixed=[3333, 3366, 6633, 6666])
        envCsOn = FlattenObservation(envCsOn)
        model = PPO.load(modelPath, env=envCsOn)        
        envCsOn = model.get_env()
        self.runEnvi(envCsOn, model, max_steps) 

    def scenario3(self, max_steps=1000):
        pass

if __name__ == '__main__':
    scenarios = Scenarios()
    steps = 10_000
    scenarios.scenario1(steps)
    scenarios.scenario2(steps)