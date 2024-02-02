from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

if __name__ == '__main__':


    env_name = 'Pendulum-v1'
    nproc = 1

    def make_env(env_id, seed):
        def _f():
            env = gym.make(env_id)
            return env
        return _f

    envs = [make_env(env_name, seed) for seed in range(nproc)]
    envs = SubprocVecEnv(envs)

    #model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=1_000_000)
    print('Done')