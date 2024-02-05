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

def train():
    current_date = datetime.now().strftime("%Y-%m-%d__%H_%M")
    total_timesteps = 0
    timesteps = 10_000
    
    load = False
    while total_timesteps < 1_000_000:
        
        # CS off
        def trainCsOff(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(total_timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(total_timesteps))

        # CS On
        def trainCsOn(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(total_timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(total_timesteps))


        argsCsOff = {
            'contentSteering': False,
            'filePrefix': 'sc1CsOff_',
            'ttl': 100,
            'shufflePrice': 99,
            'cdns': 4,
            'maxActiveClients': 20,
            'totalClients': 100,
            'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
            'cdnLocationsFixed': [3333, 3366, 6633, 6666],
            'discreteActionSpace': False,
            'bufferSize': 10,
            'clientAppearingMode': 'random'
        }    

        argsCsOn = {
            'contentSteering': True,
            'filePrefix': 'sc1CsOn_',
            'ttl': 100,
            'shufflePrice': 99,
            'cdns': 4,
            'maxActiveClients': 20,
            'totalClients': 100,
            'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
            'cdnLocationsFixed': [3333, 3366, 6633, 6666],
            'discreteActionSpace': False,
            'bufferSize': 10,
            'clientAppearingMode': 'random'
        }
        
        trainCsOff(argsCsOff, timesteps, load=load)
        trainCsOn(argsCsOn, timesteps, load=load)
        load = True
        total_timesteps += timesteps

if __name__ == '__main__':
    train()