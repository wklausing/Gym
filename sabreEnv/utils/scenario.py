from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO, A2C
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

    def scenario1(self, max_steps=100_000, mpd='sabreEnv/sabre/data/movie_597s.json'):
        '''
        Scenario 1 - VOD: 4 CDNs with constante 10 clients. Goals is to maximize reward with and without Content Steering.
        All clients fetch the same content.
        '''
        cdns = 4
        cdnLocationsFixed=[3333, 3366, 6633, 6666]
        maxActiveClients=10
        totalClients=100
        ttl=30

        # CS Off
        print('CS Off Training')
        path = 'sabreEnv/utils/data/sc1/ppo_CsOff/'
        modelPath = path + 'envCsOff_' + self.current_date
        env = GymSabreEnv(contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                               cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOff.save(modelPath)

        print('CS Off Evaluating')
        env = GymSabreEnv(contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                               cdnLocationsFixed=cdnLocationsFixed, \
                                saveData=True, savingPath=path, filePrefix='sc1_CS_Off_')
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

        # CS On
        print('CS On Training')
        env = GymSabreEnv(contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                              cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOn = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc1/ppo_CsOn/'
        modelPath = path + 'envCsOn_' + self.current_date
        modelCsOn.save(modelPath)

        print('CS On Evaluating')
        env = GymSabreEnv(contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd,  \
                               saveData=True, savingPath=path, filePrefix='sc1_CS_On_', \
                                cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
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

        # CS Off
        print('CS Off Training')
        path = 'sabreEnv/utils/data/sc2/ppo_CsOff/'
        modelPath = path + 'envCsOff_' + self.current_date
        env = GymSabreEnv(bufferSize=buffer, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOff.save(modelPath)

        print('CS Off Evaluating')
        env = GymSabreEnv(bufferSize=buffer, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed, \
                                saveData=True, savingPath=path, filePrefix='sc2_CS_Off_')
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

        # CS On
        print('CS On Training')
        env = GymSabreEnv(bufferSize=buffer, contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd, \
                              cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOn = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        path = 'sabreEnv/utils/data/sc2/ppo_CsOn/'
        modelPath = path + 'envCsOn_' + self.current_date
        modelCsOn.save(modelPath)

        print('CS On Evaluating')
        env = GymSabreEnv(bufferSize=buffer, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, ttl=ttl, mpdPath=mpd,  \
                               saveData=True, savingPath=path, filePrefix='sc2_CS_On_', \
                                cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

    def scenario3(self, max_steps=1000, mpd='sabreEnv/sabre/data/movie_597s.json'):
        '''
        Scenario 3: 4 CDNs with 10 clients. Goals is to maximize reward without costs considerations. At the end it is compared with a run that considers costs.
        '''
        cdns = 4
        cdnLocationsFixed=[3333, 3366, 6633, 6666]
        maxActiveClients=10
        totalClients=100
        ttl=30

        print('Training - Costs Off')
        path = 'sabreEnv/utils/data/sc3/ppo_noCosts/'
        modelPath = path + 'envCsOff_' + self.current_date
        env = GymSabreEnv(moneyMatters=False, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOff.save(modelPath)

        print('Evaluating - Costs Off')
        env = GymSabreEnv(moneyMatters=False, contentSteering=False, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed, \
                                saveData=True, savingPath=path, filePrefix='')
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

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
                            gridWidth=500, gridHeight=500)
        env = Monitor(env, filename=path + 'trainMonitor.csv')
        env = FlattenObservation(env)
        modelCsOff = PPO('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        modelCsOff.save(modelPath)

        print('Evaluating - Prices Shuffel')
        env = GymSabreEnv(shuffelPrice=90, contentSteering=True, cdns=cdns, maxActiveClients=maxActiveClients, totalClients=totalClients, \
                               ttl=ttl, mpdPath=mpd, cdnLocationsFixed=cdnLocationsFixed, \
                                saveData=True, savingPath=path, filePrefix='')
        env = Monitor(env, filename=path + 'evalMonitor.csv')
        env = FlattenObservation(env)
        model = PPO.load(modelPath, env=env)        
        env = model.get_env()
        self.runEnvi(env, model, max_steps)

if __name__ == '__main__':
    scenarios = Scenarios()
    steps = 100_000
    scenarios.scenario1(steps)
    scenarios.scenario2(steps)
    scenarios.scenario3(steps)
    scenarios.scenario4(steps)