from sabreEnv import GymSabreEnv
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
import multiprocessing
from datetime import datetime

def createEnv(args):
    env_name = 'gymsabre-v0'
    num_cpus = multiprocessing.cpu_count()

    def make_env(env_id):
        def _f():
            env = gym.make(env_id, **args)
            return env
        return _f

    envs = [make_env(env_name) for _ in range(num_cpus)]
    envs = SubprocVecEnv(envs)
    return envs

def train():
    current_date = datetime.now().strftime("%Y-%m-%d__%H_%M")
    timesteps = 10_000
    
        
    # CS off
    def trainCsOff(args, timesteps, load=False):
        envs = createEnv(args)
        if load:
            model = PPO.load('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))
        else:
            model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
        model.save('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOff/policyCsOff_' + str(timesteps))

    # CS On
    def trainCsOn(args, timesteps, load=False):
        envs = createEnv(args)
        if load:
            model = PPO.load('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))
        else:
            model = PPO('MlpPolicy', envs).learn(progress_bar=True, total_timesteps=timesteps)
        model.save('sabreEnv/scenarios/data/sc1/' + current_date + '/ppo_CsOn/policyCsOn_' + str(timesteps))
    

    base_args = {
        'ttl': 30,
        'shufflePrice': 60,
        'cdns': 4,
        'maxActiveClients': 20,
        'totalClients': 1000,
        'mpdPath': 'sabreEnv/sabre/data/movie_150s.json',
        'cdnLocationsFixed': [3333, 3366, 6633, 6666],
        'discreteActionSpace': False,
        'bufferSize': 25,
        'clientAppearingMode': 'random'
    }

    argsCsOff = {
        **base_args,
        'contentSteering': False,
        'filePrefix': 'sc1CsOff_',
    }

    argsCsOn = {
        **base_args,
        'contentSteering': True,
        'filePrefix': 'sc1CsOn_',
    }

    trainCsOff(argsCsOff, timesteps)
    trainCsOn(argsCsOn, timesteps)

if __name__ == '__main__':
    train()