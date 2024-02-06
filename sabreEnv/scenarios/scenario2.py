from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import time
import multiprocessing
from datetime import datetime

argsCsOff = {
    'contentSteering': True,
    'cdns': 9,
    'cdnLocationsFixed': [41583, 41749, 41915, 124583, 124749, 124915, 207583, 207749, 207915],
    'maxActiveClients': 20,
    'totalClients': 100,
    'ttl': 30,
    'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
    'gridWidth': 500, 
    'gridHeight': 500,
    'discreteActionSpace': True,
    'bufferSize': 10,
    'filePrefix': 'CsOff_',
    'verbose': True
}

argsCsOn = {
    'contentSteering': True,
    'cdns': 9,
    'cdnLocationsFixed': [41583, 41749, 41915, 124583, 124749, 124915, 207583, 207749, 207915],
    'maxActiveClients': 20,
    'totalClients': 100,
    'ttl': 30,
    'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
    'gridWidth': 500, 
    'gridHeight': 500,
    'discreteActionSpace': True,
    'bufferSize': 10,
    'filePrefix': 'CsOn_',
    'verbose': False
}

def createEnv(args):
    env_name = 'gymsabre-v0'
    num_cpus = multiprocessing.cpu_count()

    def make_env(env_id):
        def _f():
            env = gym.make(env_id, **args)
            return env
        return _f

    envs = [make_env(env_name) for _ in range(num_cpus)]
    envs = SubprocVecEnv(envs)
    return envs

def trainModel():
    current_date = datetime.now().strftime("%Y-%m-%d__%H_%M")
    total_timesteps = 0
    timesteps = 1_000
    load = False
    while total_timesteps < 1_000_000:
        
        # CS off
        def trainCsOff(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/scenarios/data/sc2/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/scenarios/data/sc2/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))

        # CS On
        def trainCsOn(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/scenarios/data/sc2/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/scenarios/data/sc2/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))

        trainCsOff(argsCsOff, timesteps, load=load)
        trainCsOn(argsCsOn, timesteps, load=load)
        load = True
        total_timesteps += 100_000

def evalModel(args, path):
    args['saveData'] = True
    env = gym.make('gymsabre-v0', **args)
    model = PPO.load(path, env=env, verbose=1)
    vec_env = model.get_env()
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)


if __name__ == '__main__':
    '''
    Scenario X: Environment with 9 CDNs, 100 acticeCients, 60s video, 30s TTL, 10 buffer size
    '''
    trainModel()
    # pathCsOff = '/Users/prabu/Desktop/sc1/2024-02-04__08_58/ppo_CsOff/policyCsOff_100000'
    # pathCsOn = '/Users/prabu/Desktop/sc1/2024-02-04__08_58/ppo_CsOff/policyCsOn_100000'
    # evalModel(argsCsOff, pathCsOff)
    # evalModel(argsCsOn, pathCsOn)
