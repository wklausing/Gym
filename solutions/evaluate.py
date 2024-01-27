from sabreEnv import GymSabreEnv, SabreActionWrapper
from gymnasium.wrappers import FlattenObservation
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
import os
import glob


class Evaluater:

    def __init__(self):
        self.env_name = 'gymsabre-v0'
        self.env = gym.make(self.env_name, cdns=4, maxActiveClients=10, totalClients=10)
        self.env = FlattenObservation(self.env)

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
        model = PPO.load(latestModel, env=self.env) 
        return model
    
    def loadA2C(self, path=None):
        if path is None:
            latestModel = self.load_latest_model(algorithm='a2c')
        model = A2C.load(latestModel, env=self.env) 
        return model

    def evaluate(self, model):
        evaluate = evaluate_policy(model, self.env)
        print('Evaluation result:' + str(evaluate))

    def runEnv(self, model, steps=1_000):
        env = model.get_env()
        obs = env.reset()
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            observation, reward, terminated, info = env.step(action)
            if terminated:
                observation = env.reset()
        env.close()

if __name__ == '__main__':
    evaluater = Evaluater()
    model = evaluater.loadPPO()
    evaluater.evaluate(model)