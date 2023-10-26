from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray import air
from ray import tune

config = DQNConfig()
config = config.training( 
    num_atoms=tune.grid_search(list(range(1,11))))
config = config.environment(env="CartPole-v1") 
tune.Tuner(  
    "DQN",
    run_config=air.RunConfig(stop={"episode_reward_mean":200}),
    param_space=config.to_dict()
).fit()