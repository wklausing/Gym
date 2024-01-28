

from matplotlib import pyplot as plt
import pandas as pd


class Visualize:

    def __init__(self):
        pass

    def plot_combined_reward_development(self, dataLocation1='sabreEnv/utils/data/sc1/ppo_CsOff/sc1_CS_Off_cpData.csv', \
                                         dataLocation2='sabreEnv/utils/data/sc1/ppo_CsOn/sc1_CS_On_cpData.csv', episode_number=1):
        """
        Plots the development of the reward for a given episode for two datasets on the same plot.
        The x-axis is time, and the y-axis is the reward.

        :param data1: DataFrame containing the first dataset.
        :param data2: DataFrame containing the second dataset.
        :param episode_number: The episode number for which to plot the reward.
        """
        # Load data
        data1 = pd.read_csv(dataLocation1)
        data2 = pd.read_csv(dataLocation2)

        # Filter data for the specified episode in both datasets
        episode_data1 = data1[data1['episode'] == episode_number]
        episode_data2 = data2[data2['episode'] == episode_number]

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(episode_data1['time'], episode_data1['reward'], label='Reward (Content Steering Off)')
        plt.plot(episode_data2['time'], episode_data2['reward'], label='Reward (Content Steering On)')
        plt.title(f'Reward Development Comparison in Episode {episode_number}')
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    vis = Visualize()
    # Example usage: plot reward development comparison for the first episode
    vis.plot_combined_reward_development()