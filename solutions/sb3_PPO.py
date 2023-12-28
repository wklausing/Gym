from sabreEnv import GymSabreEnv, SabreActionWrapper
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation


print('Setup environment')

env_name = "gymsabre-v0"
env = gym.make(env_name, gridSize=100*100, serviceLocations=4, clients=10, saveData=False)
env = FlattenObservation(env)

print('Start training')

model = PPO("MlpPolicy", env).learn(total_timesteps=300_000, progress_bar=True)
model.save("solutions/policies/ppo_gymsabre-v1")

print('Finished training')

print('Start using model')

model = PPO.load('solutions/policies/ppo_gymsabre-v1', env=env) 
vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, info = vec_env.step(action)

    if terminated:
        observation = vec_env.reset()

    print(reward)
env.close()

print('Finished using model')