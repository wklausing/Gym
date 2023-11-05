from sabreEnv import GymSabreEnv, SabreActionWrapper
import gymnasium as gym
# from stable_baselines3 import A2C
# from gymnasium.wrappers import FlattenObservation

import itertools
import random


# print('Start training')
clients = 10
edgeServers = 4
action_space = gym.spaces.MultiDiscrete(clients * edgeServers * [edgeServers])
print(action_space.sample())


