from sabreEnv import GymSabreEnv
import gymnasium as gym

import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import tune


env = gym.make("gymsabre-v0")
env_name = "gymsabre-v0"
tune.register_env(env_name, lambda cfg: env)

config = (
    DQNConfig()
    .environment(env_name)
    .training()
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=3)
)

algo = config.build()

print("3. train it")
for _ in range(100):
    algo.train()
    print(algo.train())
    print(f"Training at {_}") 

print("4. and evaluate it")
#algo.evaluate() 
checkpoint_dir = algo.save()
print(f"Checkpoint saved in directory {checkpoint_dir}")