from sabreEnv import GymSabreEnv
import gymnasium as gym

args = {
    'contentSteering': False,
    'filePrefix': 'random_',
    'ttl': 100,
    'shufflePrice': 99,
    'cdns': 4,
    'maxActiveClients': 20,
    'totalClients': 100,
    'mpdPath': 'sabreEnv/sabre/data/movie_60s.json',
    'cdnLocationsFixed': [3333, 3366, 6633, 6666],
    'discreteActionSpace': True,
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