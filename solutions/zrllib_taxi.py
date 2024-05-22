from ray.rllib.algorithms.ppo import PPOConfig

env_name = "Taxi-v3"

config = (  # 1. Configure the algorithm,
    PPOConfig()
    .environment(env_name)
    .rollouts(num_rollout_workers=2)
    .framework("tf2")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()  # 2. build the algorithm,

for _ in range(100):
    algo.train()
    print(algo.train())  # 3. train it,

algo.evaluate()  # 4. and evaluate it.
checkpoint_dir = algo.save()
print(f"Checkpoint saved in directory {checkpoint_dir}")
