import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from sabreEnv import GymSabreEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == '__main__':
    env_name = 'gymsabre-v0'

    # gym.register(
    #     id="gymsabre-v0",
    #     entry_point="sabreEnv:GymSabreEnv",
    # )   

    vec_env = make_vec_env('gymsabre-v0', n_envs=4)

    model = PPO("MultiInputPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=25000)
    
    # nproc = 1

    # def make_env(env_id, seed):
    #     def _f():
    #         env = gym.make(env_id)
    #         env = FlattenObservation(env)
    #         return env
    #     return _f

    # envs = [make_env(env_name, seed) for seed in range(nproc)]
    # envs = SubprocVecEnv(envs)

    # #model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=1_000_000)
    # print('Done')
