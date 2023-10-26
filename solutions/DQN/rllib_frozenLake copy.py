import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn import DQNConfig

from sabreEnv import GymSabreEnv
from ray import tune

from gymnasium.wrappers import FlattenObservation

env_name = "gymsabre-v0"
env = gym.make("gymsabre-v0")
print(env.observation_space.sample())
env = FlattenObservation(env)
print('Here is the new observation space:')
print(env.observation_space.sample())




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