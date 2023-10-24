import sabre
import gymnasium as gym

env = gym.make('sabre/Sabre-v0')

print(env.action_space.sample())