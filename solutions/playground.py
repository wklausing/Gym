from sabreEnv import GymSabreEnv
import gymnasium as gym

foo = GymSabreEnv().action_space.sample()
print(foo)


env = gym.make("gymsabre-v0")
print(env.reset())
