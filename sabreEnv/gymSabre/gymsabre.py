import numpy as np

import gymnasium as gym
from gymnasium import spaces

class GymSabreEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Env variables
    grid_size = 100 * 100
    time = np.array(0, dtype='int')

    # CP-Agent variables
    money = np.array(100, dtype='int')

    # Client variables
    clientCount = 10
    clientsLocations = np.ones(clientCount, dtype=int)
    clients = []

    # CDN variables
    edgeServerCount = 4
    edgeServerLocations = np.ones(edgeServerCount, dtype=int)
    edgeServerPrices = np.ones(edgeServerCount, dtype=int)
    edgeServers = []

    def __init__(self, render_mode=None):
        # Observation space for CP agent. Contains location of clients, location of edge-servers, pricing of edge-server, and time in seconds.        
        self.observation_space = spaces.Dict(
            {
                'clientsLocations': gym.spaces.MultiDiscrete([self.grid_size] * self.clientCount),
                'edgeServerLocations': gym.spaces.MultiDiscrete([self.grid_size ] * self.edgeServerCount),
                'edgeServerPrices': spaces.Box(0, 10, shape=(self.edgeServerCount,), dtype=float),
                'time': spaces.Box(0, 100_000, shape=(1,), dtype=int),
                'money': spaces.Box(0, 100_000, shape=(1,), dtype=int),
            }
        )

        # Action space contains ability to buy contigent from edge-server and to steer client to another edge-server.
        # self.action_space = spaces.Dict(
        #     {
        #         'buyContigent': gym.spaces.MultiDiscrete([self.grid_size] * self.edgeServerCount),
        #         'steerClient': gym.spaces.MultiDiscrete([self.edgeServerCount] * self.clientCount),
        #     }
        # )

        self.buyContingent = [100] * self.edgeServerCount
        self.steerClient = [self.edgeServerCount] * self.clientCount
        self.action_space = gym.spaces.MultiDiscrete(self.buyContingent + self.steerClient)

    def _get_obs(self):
        return {"clientsLocations": self.clientsLocations, "edgeServerLocations": self.edgeServerLocations
                , "edgeServerPrices": self.edgeServerPrices, "time": self.time
                , "money": self.money}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset variables
        self.time = np.array([0], dtype='int')
        self.money = np.array([100_000], dtype='int')

        # Reset clients
        self.clientsLocations = self.np_random.integers(0, self.grid_size, size=self.clientCount, dtype=int)
        self.clients = []
        for c in range(self.clientCount):
            self.clients.append(Client(self.clientsLocations[c]))

        # Reset edge servers
        self.edgeServerLocations = self.np_random.integers(0, self.grid_size, size=self.edgeServerCount, dtype=int)
        self.edgeServerPrices = np.round(np.random.uniform(0, 10, size=self.edgeServerCount), 2)
        self.edgeServers = []
        for e in range(self.edgeServerCount):
            self.edgeServers.append(EdgeServer(self.edgeServerLocations[e], self.edgeServerPrices[e]))

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        #return self.state, reward, terminated, truncated, {}

    def step(self, action):
        time = self.time.item()
        money = self.money.item()
        
        reward = 0

        # An episode is done when the CP is out of money or time is over
        terminated = money <= 0 or time >= 100_000

        if not terminated:
            # Buy contigent
            #buyContigent = action['buyContigent']
            buyContigent = action[:len(self.buyContingent)] 

            for index, edgeServer in enumerate(self.edgeServers):
                money = edgeServer.sellContigent(money, buyContigent[index])
            
            # Steer clients
            #steerClient = action['steerClient']
            steerClient = action[len(self.buyContingent):] 
            for index, client in enumerate(self.clients):
                client.edgeServer = self.edgeServers[steerClient[index]]
                money += client.fetchContent()
                reward += client.fetchContent()

        observation = self._get_obs()
        info = self._get_info()

        time += 1

        self.time = np.array([time], dtype='int')
        self.money = np.array([money], dtype='int')

        return observation, reward, terminated, False, info


class Client:

    def __init__(self, location):
        self.location = location

    def fetchContent(self):
        if self._edgeServer.soldContigent > 0:
            self._edgeServer.soldContigent -= 1
            return 1
        else: 
            return -1

    @property
    def edgeServer(self):
        return self._edgeServer
    
    @edgeServer.setter
    def edgeServer(self, edgeServer):
        self._edgeServer = edgeServer


class EdgeServer:

    def __init__(self, location, price=1):
        self.location = location
        self.price = price
        self.soldContigent = 0

    def sellContigent(self, cpMoney, amount):
        leftOverMoney = 0
        if cpMoney >= amount * self.price:
            cpMoney -= amount * self.price
            self.soldContigent += amount
            leftOverMoney = amount * self.price
        else:
            leftOverMoney = cpMoney
        return round(leftOverMoney, 2)


if __name__ == "__main__":
    print('Start!')
    env = GymSabreEnv(render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    print('Done!')
