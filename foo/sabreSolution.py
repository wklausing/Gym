import gymnasium as gym
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from gymnasium.envs.registration import register


register(
    id="sabre",
    entry_point="gymnasium.envs.classic_control.cartpole:CartPoleEnv",
    vector_entry_point="gymnasium.envs.classic_control.cartpole:CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)




env_name = "Taxi-v3"

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

print("4. and evaluate it")
#algo.evaluate() 
checkpoint_dir = algo.save()
print(f"Checkpoint saved in directory {checkpoint_dir}")

env = gym.make(env_name, render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = algo.compute_single_action(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()