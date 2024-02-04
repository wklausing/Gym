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
    'contentSteering': False,
    'ttl': 100,
    'shufflePrice': 99,
    'cdns': 4,
    'maxActiveClients': 20,
    'totalClients': 1000,
    'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
    'cdnLocationsFixed': [3333, 3366, 6633, 6666],
    'discreteActionSpace': True,
    'bufferSize': 10,
    'filePrefix': 'CsOff_',
    'verbose': False
}

argsCsOn = {
    'contentSteering': True,
    'ttl': 100,
    'shufflePrice': 99,
    'cdns': 4,
    'maxActiveClients': 20,
    'totalClients': 1000,
    'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
    'cdnLocationsFixed': [3333, 3366, 6633, 6666],
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
    current_date = 'foo'#datetime.now().strftime("%Y-%m-%d__%H_%M")
    total_timesteps = 0
    timesteps = 1_000
    load = False
    while total_timesteps < 1_000_000:
        
        # CS off
        def trainCsOff(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))

        # CS On
        def trainCsOn(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))

        trainCsOff(argsCsOff, timesteps, load=load)
        trainCsOn(argsCsOn, timesteps, load=load)
        load = True
        total_timesteps += 100_000
        break

def evalModel(args, path):
    args['saveData'] = True
    env = gym.make('gymsabre-v0', **args)
    model = PPO.load(path, env=env, verbose=1)
    vec_env = model.get_env()
    vec_env.seed(1)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward}, Std reward: {std_reward}')

if __name__ == '__main__':
    trainModel()
    pathCsOff = 'sabreEnv/scenarios/data/sc1/foo/ppo_CsOff/policyCsOff_1000'
    pathCsOn = 'sabreEnv/scenarios/data/sc1/foo/ppo_CsOn/policyCsOn_1000'
    evalModel(argsCsOff, pathCsOff)
    evalModel(argsCsOn, pathCsOn)
