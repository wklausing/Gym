from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO

args = {
    'contentSteering': False,
    'filePrefix': 'random_',
    'ttl': 100,
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


env_name = 'gymsabre-v0'
env = gym.make(env_name, **args)
obs = env.reset(seed=42)
for i in range(1_000):
    action = env.action_space.sample()#model.predict(obs, deterministic=True)
    obs, rewards, done, truncated, info = env.step(action)
env.close()


args['filePrefix'] = 'CsOff_'
env = gym.make(env_name, **args)
pathCsOff = '/Users/prabu/Desktop/2024-02-06__00_04/ppo_CsOff/policyCsOff_0'
model = PPO.load(pathCsOff, env=env) 
vec_env = model.get_env()
vec_env.seed(42)
obs = vec_env.reset()
for _ in range(1_000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, info = vec_env.step(action)
    if terminated:
        observation = vec_env.reset()
env.close()


args['contentSteering'] = True
args['filePrefix'] = 'CsOn_'
env = gym.make(env_name, **args)
pathCsOn = '/Users/prabu/Desktop/2024-02-06__00_04/ppo_CsOn/policyCsOn_0'
model = PPO.load(pathCsOn, env=env) 
vec_env = model.get_env()
vec_env.seed(42)
obs = vec_env.reset()
for _ in range(1_000):
    action, _ = model.predict(obs, deterministic=True)
    observation, reward, terminated, info = vec_env.step(action)
    if terminated:
        observation = vec_env.reset()
env.close()