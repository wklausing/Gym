from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor

print('Setup environment')

env_name = "gymsabre-v0"
env = gym.make(env_name, cdns=4, maxActiveClients=10, totalClients=100)
# env = Monitor(env, filename='./monitor.csv')
env = FlattenObservation(env)


print('Start training')

# model = PPO("MlpPolicy", env).learn(progress_bar=True, total_timesteps=100)
# model.save("solutions/policies/ppo_gymsabre-v1")

print('Finished training')

print('Start using model')

model = PPO.load('solutions/policies/ppo_gymsabre-v1', env=env) 
env = model.get_env()
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, info = env.step(action)

    if terminated:
        observation = env.reset()
env.close()

print('Finished using model')