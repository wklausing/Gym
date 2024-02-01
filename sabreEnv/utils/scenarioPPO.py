from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from datetime import datetime

class Scenarios:
    '''
    Serves merely as a note for scenarios and their respective parameters.
    '''
    def __init__(self):
        self.current_date = datetime.now().strftime("%Y-%m-%d__%H_%M")
        self.env_name = 'gymsabre'

    def runEnvi(self, env, model, max_steps=1000):
        obs = env.reset()
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            observation, reward, terminated, info = env.step(action)
            if terminated:
                observation = env.reset()
        env.close()

    def scenario1(self, max_steps=100_000, mpd='sabreEnv/sabre/data/movie_597s.json', path='sabreEnv/utils/data/sc1/', weightCost=1):
        '''
        Scenario 1 - VOD: 4 CDNs with constante 10 clients. Goals is to maximize reward with and without Content Steering.
        All clients fetch the same content.
        '''
        cdns = 4
        cdnLocationsFixed=[3333, 3366, 6633, 6666]
        maxActiveClients=10
        totalClients=100
        ttl=30
        path = path + self.current_date

        print('CS Off Training')
        pathCsOff = path + '/ppo_CsOff/'
        modelCsOffPath = pathCsOff + 'policyCsOff'
        env = GymSabreEnv(contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                               cdnLocationsFixed=cdnLocationsFixed, weightCost=weightCost, saveData=False, dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOff + 'trainCsOff')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOff.save(modelCsOffPath)

        print('CS On Training')
        pathCsOn = path + '/ppo_CsOn/'
        modelCsOnPath = pathCsOn + 'policyCsOn'
        env = GymSabreEnv(contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                              cdnLocationsFixed=cdnLocationsFixed, weightCost=weightCost, saveData=False, dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOn + 'trainCsOn')
        env = FlattenObservation(env)
        modelCsOn = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOn.save(modelCsOnPath)


        print('CS Off Evaluating')
        env = GymSabreEnv(contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                            cdnLocationsFixed=cdnLocationsFixed, weightCost=weightCost, \
                                saveData=True, savingPath=pathCsOff, filePrefix='', dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOff + 'evalCsOff')
        env = FlattenObservation(env)
        model = PPO.load(modelCsOffPath, env=env)
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

        print('CS On Evaluating')
        env = GymSabreEnv(contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd,  \
                            cdnLocationsFixed=cdnLocationsFixed, weightCost=weightCost, \
                            saveData=True, savingPath=pathCsOn, filePrefix='', dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOn + 'evalCsOn')
        env = FlattenObservation(env)
        model = PPO.load(modelCsOnPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

    def scenario2(self, max_steps=100_000, mpd='sabreEnv/sabre/data/movie_597s.json'):
        '''
        Scenario 2: 4 CDNs with 10 clients. Buffer is 5 seconds. Goals is to maximize reward with and without Content Steering.
        '''
        cdns = 4
        cdnLocationsFixed=[3333, 3366, 6633, 6666]
        maxActiveClients=10
        totalClients=100
        ttl=30
        buffer = 5
        path = 'sabreEnv/utils/data/sc2/' + self.current_date

        # CS Off
        print('CS Off Training')
        pathCsOff = path + '/ppo_CsOff/'
        env = GymSabreEnv(bufferSize=buffer, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed, dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOff + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOffPath = pathCsOff + 'sc2PolicyCsOff'
        modelCsOff.save(modelCsOffPath)

        print('CS On Training')
        pathCsOn = path + '/ppo_CsOn/'
        env = GymSabreEnv(bufferSize=buffer, contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                              cdnLocationsFixed=cdnLocationsFixed, dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOn + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOn = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc2/ppo_CsOn/'
        modelCsOnPath = pathCsOn + 'sc2PolicyCsOn'
        modelCsOn.save(modelCsOnPath)

        print('CS Off Evaluating')
        env = GymSabreEnv(bufferSize=buffer, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed, \
                                saveData=True, savingPath=path, filePrefix='sc2_CS_Off_', dqnActionSpace=False)
        env = Monitor(env, filename=pathCsOff + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelCsOffPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

        print('CS On Evaluating')
        env = GymSabreEnv(bufferSize=buffer, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd,  \
                               saveData=True, savingPath=path, filePrefix='sc2_CS_On_', \
                                cdnLocationsFixed=cdnLocationsFixed, dqnActionSpace=False)
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelCsOnPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

    def scenario3(self, max_steps=1000, mpd='sabreEnv/sabre/data/movie_597s.json'):
        '''
        Scenario 3: 4 CDNs with 10 clients. Goals is to maximize reward without costs considerations. At the end it is compared with a run that considers costs.
        '''
        self.scenario1(max_steps=max_steps, mpd=mpd, path='sabreEnv/utils/data/sc3/', weightCost=0)

    def scenario4(self, max_steps=1000, mpd='sabreEnv/sabre/data/movie_597s.json'):
        '''
        Scenario 4: 4 CDNs with 10 clients. Goals is to maximize reward, while prices are changing over time.
        '''
        cdns = 9
        cdnLocationsFixed=[41583, 41749, 41915, 124583, 124749, 124915, 207583, 207749, 207915]
        maxActiveClients=16
        totalClients=100
        ttl=30

        print('Training - Prices Shuffel')
        path = 'sabreEnv/utils/data/sc4/ppo_pricesShuffel/'
        modelPath = path + 'envCsOff_' + self.current_date
        env = GymSabreEnv(shuffelPrice=90, contentSteering=True, cdns=cdns, \
                          maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                            gridWidth=500, gridHeight=500, dqnActionSpace=False)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOff.save(modelPath)

        print('Evaluating - Prices Shuffel')
        env = GymSabreEnv(shuffelPrice=90, contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed, \
                                saveData=True, savingPath=path, filePrefix='', dqnActionSpace=False)
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

if __name__ == '__main__':
    scenarios = Scenarios()
    steps = 10_000
    scenarios.scenario1(max_steps=steps)
    # scenarios.scenario2(max_steps=steps)
    # scenarios.scenario3(max_steps=steps)
    # scenarios.scenario4(max_steps=steps)