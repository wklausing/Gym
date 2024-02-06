import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
cs_off_data = pd.read_csv('sabreEnv/gymSabre/data/CsOff_cpData.csv')
cs_on_data = pd.read_csv('sabreEnv/gymSabre/data/CsOn_cpData.csv')
random_data = pd.read_csv('sabreEnv/gymSabre/data/random_cpData.csv')

# Filter rows where episode == 1 for each dataset
cs_off_episode_1 = cs_off_data[cs_off_data['episode'] == 1]
cs_on_episode_1 = cs_on_data[cs_on_data['episode'] == 1]
random_episode_1 = random_data[random_data['episode'] == 1]

# Plot for qoeNorm
plt.figure(figsize=(10, 6))
plt.plot(cs_off_episode_1['time'], cs_off_episode_1['qoeNorm'], label='CS Off')
plt.plot(cs_on_episode_1['time'], cs_on_episode_1['qoeNorm'], label='CS On')
plt.plot(random_episode_1['time'], random_episode_1['qoeNorm'], label='Random')
plt.xlabel('Time')
plt.ylabel('qoeNorm')
plt.title('QoE Normalized over Time by File')
plt.legend()
plt.show()

# Plot for reward
plt.figure(figsize=(10, 6))
plt.plot(cs_off_episode_1['time'], cs_off_episode_1['reward'], label='CS Off')
plt.plot(cs_on_episode_1['time'], cs_on_episode_1['reward'], label='CS On')
plt.plot(random_episode_1['time'], random_episode_1['reward'], label='Random')
plt.xlabel('Time')
plt.ylabel('Reward')
plt.title('Reward over Time by File')
plt.legend()
plt.show()
