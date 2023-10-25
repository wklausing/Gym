from sabreEnv import GymSabreEnv
import gymnasium as gym

foo = GymSabreEnv().action_space.sample()
print(foo)



env = gym.make("gymsabre-v0")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    #print(observation, reward, terminated, truncated, info)
    if reward != 10: print(reward)

env.close()
