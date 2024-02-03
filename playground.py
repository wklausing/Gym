from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import time
import multiprocessing


if __name__ == '__main__':
    start_time = time.time()
    env_name = 'gymsabre-v0'
    num_cpus = multiprocessing.cpu_count()

    def make_env(env_id, seed):
        def _f():
            env = gym.make(env_id)
            return env
        return _f

    envs = [make_env(env_name, seed) for seed in range(num_cpus)]
    envs = SubprocVecEnv(envs)

    model = PPO('MultiInputPolicy', envs).learn(progress_bar=True, total_timesteps=1_000)
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time} seconds")

    #Steps: 16,384 Time elapsed: 476.4250690937042 seconds
    #Steps: 2,048  Time elapsed: 111.65557193756104 seconds
    #Steps: 8x     Time elapsed: 4x seconds