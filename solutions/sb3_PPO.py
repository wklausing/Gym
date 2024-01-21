from sabreEnv import GymSabreEnv, SabreActionWrapper
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation


print('Setup environment')

env_name = "gymsabre-v0"
env = gym.make(env_name, cdnLocations=4, maxActiveClients=10, totalClients=100, saveData=True, render_mode='human')
env = FlattenObservation(env)

# print('Start training')

# model = PPO("MlpPolicy", env).learn(total_timesteps=1_000, progress_bar=True)
# model.save("solutions/policies/ppo_gymsabre-v1")

# print('Finished training')

# print('Start using model')

model = PPO.load('solutions/policies/ppo_gymsabre-v1', env=env) 
env = model.get_env()
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, info = env.step(action)

    if terminated:
        observation = env.reset()

    print(reward)
env.close()

# print('Finished using model')