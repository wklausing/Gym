from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO

env_name = 'gymsabre-v0'
args = {
    'contentSteering': False,
    'ttl': 30,
    'shufflePrice': 99,
    'cdns': 4,
    'maxActiveClients': 20,
    'totalClients': 100,
    'mpdPath': 'sabreEnv/sabre/data/movie_150s.json',
    'cdnLocationsFixed': [3333, 3366, 6633, 6666],
    'discreteActionSpace': False,
    'bufferSize': 10,
    'verbose': False,
    'saveData': True,
    'clientAppearingMode': 'random'
}


args['filePrefix'] = 'RandomCsOff_'
env = gym.make(env_name, **args)
seed = 42
obs = env.reset(seed=seed)
for i in range(1_000):
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        seed += 1
        observation, info = env.reset()
env.close()


args['filePrefix'] = 'RandomCsOn_'
args['contentSteering'] = True
env = gym.make(env_name, **args)
seed = 42
obs = env.reset(seed=seed)
for i in range(1_000):
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        seed += 1
        observation, info = env.reset()
env.close()


args['filePrefix'] = 'CsOff_'
pathCsOff = '/Users/prabu/Desktop/2024-02-06__00_04/ppo_CsOff/policyCsOff_0'
model = PPO.load(pathCsOff) 
env = gym.make(env_name, **args)
seed = 42
obs, _ = env.reset(seed=seed)
for _ in range(1_000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        seed += 1
        observation, info = env.reset()
env.close()


args['contentSteering'] = True
args['filePrefix'] = 'CsOn_'
pathCsOn = '/Users/prabu/Desktop/2024-02-06__00_04/ppo_CsOn/policyCsOn_0'
model = PPO.load(pathCsOn) 
env = gym.make(env_name, **args)
seed = 42
obs, _ = env.reset(seed=seed)
for _ in range(1_000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        seed += 1
        observation, info = env.reset()
env.close()
