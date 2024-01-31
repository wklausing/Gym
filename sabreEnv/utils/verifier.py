from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from gymnasium.wrappers import FlattenObservation
from datetime import datetime
from stable_baselines3.common.monitor import Monitor

def runEnvi(mpd='sabreEnv/sabre/data/movie_597s.json'):
    env_name = "gymsabre-v0"
    env = gym.make(env_name, cdns=4, maxActiveClients=10, totalClients=100, contentSteering=True, ttl=30, mpdPath=mpd, \
                               cdnLocationsFixed=[3333, 3366, 6633, 6666])
    env = Monitor(env, filename='./monitor.csv')
    env = FlattenObservation(env)

    print('Start using model')

    model = PPO.load('/Users/prabu/Desktop/sc1/ppo_CsOn/envCsOn_2024-01-30.zip', env=env) 
    env = model.get_env()
    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        observation, reward, terminated, info = env.step(action)

        if terminated:
            observation = env.reset()
    env.close()

# mpd='sabreEnv/sabre/data/movie_597s.json'
# env = GymSabreEnv(cdns=4, maxActiveClients=10, totalClients=100, contentSteering=False, ttl=30, mpdPath=mpd, \
#                                cdnLocationsFixed=[3333, 3366, 6633, 6666])
# env = FlattenObservation(env)
# model = PPO.load('/Users/prabu/Desktop/sc1/ppo_CsOff/envCsOff_2024-01-30.zip', env=env)
# env = model.get_env()
# runEnvi(env, model)

runEnvi()