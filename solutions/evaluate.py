from sabreEnv import GymSabreEnv
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import os
import glob


class Evaluater:

    def __init__(self):
        self.argsCsOff = {
            'contentSteering': False,
            'filePrefix': 'CsOff_',
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
        self.env_name = 'gymsabre-v0'
        self.env = gym.make(self.env_name, **self.argsCsOff)
        #self.env = FlattenObservation(self.env)
        pass

    def load_latest_model(self, directory='solutions/policies', algorithm='ppo', env_name='gymsabre'):
        search_pattern = os.path.join(directory, f'{algorithm}_{env_name}_*.zip')
        model_files = glob.glob(search_pattern)
        if not model_files:
            raise FileNotFoundError('No model files found in the specified directory.')
        latest_model = sorted(model_files, reverse=True)[0]
        latest_model = latest_model[:-4]
        return latest_model
    
    def loadPPO(self, path=None):
        if path is None:
            latestModel = self.load_latest_model(algorithm='ppo')
        # latestModel = '/Users/prabu/Desktop/100k/ppo_CsOff/policyCsOff_100000'
        latestModel = '/Users/prabu/Desktop/100k/ppo_CsOff/policyCsOff_100000'
        model = PPO.load(latestModel, env=self.env) 
        return model
    
    def loadA2C(self, path=None):
        if path is None:
            latestModel = self.load_latest_model(algorithm='a2c')
        model = A2C.load(latestModel, env=self.env) 
        return model

    def evaluate(self, model):
        model.seed(1)
        evaluate = evaluate_policy(model, self.env, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
        print('Evaluation result:' + str(evaluate))

    def runEnv(self, model, args, steps=1_000):
        env = gym.make('gymsabre-v0', **args)
        modelPath = '/Users/prabu/Desktop/100k/ppo_CsOff/policyCsOff_100000'
        model = PPO.load(modelPath, env=env) 
        vec_env = model.get_env()
        obs = vec_env.seed(1)
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            observation, reward, terminated, info = vec_env.step(action)
            if terminated:
                observation = vec_env.reset()
        self.env.close()


# args['saveData'] = True
# env = gym.make('gymsabre-v0', **args)
# model = PPO.load(path, env=env, verbose=1)
# vec_env = model.get_env()
# vec_env.seed(1)
# mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
# print(f'Mean reward: {mean_reward}, Std reward: {std_reward}')

if __name__ == '__main__':

    argsCsOff = {
        'contentSteering': False,
        'filePrefix': 'CsOff_',
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

    evaluater = Evaluater()
    model = evaluater.loadPPO()
    evaluater.runEnv(model, argsCsOff)
    #evaluater.evaluate(model)