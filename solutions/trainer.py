from sabreEnv import GymSabreEnv, SabreActionWrapper
import gymnasium as gym
from stable_baselines3 import PPO, A2C
from gymnasium.wrappers import FlattenObservation
from datetime import datetime


class Trainer:

    def __init__(self):
        self.current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.env_name = 'gymsabre'
        self.env = gym.make(self.env_name, cdnLocations=4, maxActiveClients=10, totalClients=10)
        self.env = FlattenObservation(self.env)

    def ppoTrainer(self, env, max_steps=10_000):
        model = PPO('MlpPolicy', env, n_steps=max_steps).learn(total_timesteps=max_steps, progress_bar=True)
        model.save('solutions/policies/ppo_' + self.env_name + '_' + self.current_date)

    def a2cTrainer(self, env, max_steps=10_000):
        model = A2C('MlpPolicy', env).learn(total_timesteps=max_steps, progress_bar=True)
        model.save('solutions/policies/a2c_' + self.env_name + '_' + self.current_date)


if __name__ == '__main__':    
    trainer = Trainer()
    trainer.ppoTrainer(trainer.env, max_steps=100)