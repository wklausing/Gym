# from sabreEnv import GymSabreEnv
# import gymnasium as gym
# from stable_baselines3 import PPO
# from gymnasium.wrappers import FlattenObservation
# from stable_baselines3.common.monitor import Monitor

# print('Setup environment')

# env_name = "gymsabre-v0"
# env = gym.make(env_name, cdns=4, maxActiveClients=10, totalClients=100)
# # env = Monitor(env, filename='./monitor.csv')
# env = FlattenObservation(env)


# print('Start training')

# # model = PPO("MlpPolicy", env).learn(progress_bar=True, total_timesteps=100)
# # model.save("solutions/policies/ppo_gymsabre-v1")

# print('Finished training')

# print('Start using model')

# model = PPO.load('solutions/policies/ppo_gymsabre-v1', env=env) 
# env = model.get_env()
# obs = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     observation, reward, terminated, info = env.step(action)

#     if terminated:
#         observation = env.reset()
# env.close()

# print('Finished using model')


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
file_paths = [
    "sabreMetrics50.csv",
    "sabreMetrics500.csv",
    "sabreMetrics5000.csv"
]

# Store the data from each file along with its bandwidth
data = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    bandwidth = file_path.split('sabreMetrics')[-1].split('.csv')[0]
    data.append((bandwidth, df))

print(data[1][1]['time_average_score'])

# Prepare to plot
#plt.figure(figsize=(12, 6))

# Loop through each dataset and plot
for bandwidth, df in data:
    # Filter rows where status is 'downloadSegment'
    filtered_df = df[df['status'] == 'downloadSegment']

    print(filtered_df['time_average_score'].values)
    

    # Plot time_average_score against the count of occurrences
#     plt.plot(filtered_df['time_average_score'].values, label=f'Bandwidth {bandwidth} kbps')

# # Setting up the plot
# plt.xlabel('Count where status is "downloadSegment"')
# plt.ylabel('Time Average Score')
# plt.title('Time Average Score for Different Bandwidths')
# plt.legend()

# Show the plot
#plt.show()
