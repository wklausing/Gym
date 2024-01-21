from sabreEnv import GymSabreEnv, SabreActionWrapper
import gymnasium as gym
from stable_baselines3 import A2C
from gymnasium.wrappers import FlattenObservation


print('Setup environment')

env_name = "gymsabre-v0"
env = gym.make(env_name, cdnLocations=4, maxActiveClients=10, totalClients=100, saveData=True, render_mode='human')
env = FlattenObservation(env)

print('Start training')

# model = A2C("MlpPolicy", env).learn(total_timesteps=10_000, progress_bar=True)
# model.save("solutions/policies/a2c_gymsabre-v1_31")

print('Finished training')

# print('Start using model')

model = A2C.load('solutions/policies/a2c_gymsabre-v1_31', env=env) 
vec_env = model.get_env()
obs = vec_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, info = vec_env.step(action)

    if terminated:
        observation = vec_env.reset()

    print(reward)
env.close()

# print('Finished using model')