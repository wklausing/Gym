from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
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


if __name__ == '__main__':
    start_time = time.time()
    current_date = datetime.now().strftime("%Y-%m-%d__%H_%M")

    argsCsOff = {
        'contentSteering': False,
        'cdns': 4,
        'maxActiveClients': 10,
        'totalClients': 100,
        'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
        'cdnLocationsFixed': [3333, 3366, 6633, 6666],
        'discreteActionSpace': True,
        'bufferSize': 10
    }

    argsCsOn = {
        'contentSteering': True,
        'cdns': 4,
        'maxActiveClients': 10,
        'totalClients': 100,
        'ttl': 30,
        'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
        'cdnLocationsFixed': [3333, 3366, 6633, 6666],
        'discreteActionSpace': True,
        'bufferSize': 10
    }

    total_timesteps = 0
    timesteps = 100_000
    load = False
    while total_timesteps < 1_000_000:
        
        # CS off
        def train(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/utils/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/utils/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))

        # CS On
        def train(args, timesteps, load=False):
            envs = createEnv(args)
            if load:
                model = PPO.load('sabreEnv/utils/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))
            else:
                model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
            model.save('sabreEnv/utils/data/sc1/' + current_date + '/ppo_CsOff/policyCsOn_' + str(timesteps))

        train(argsCsOff, timesteps, load=load)
        train(argsCsOn, timesteps, load=load)
        load = True
        total_timesteps += 100_000