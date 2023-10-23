import numpy as np

import gymnasium as gym
from gymnasium import spaces


class Sabre(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # Variables
    grid_size = 100 * 100
    time = 0

    # CP-Agent variables
    money = 100

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
        self.action_space = spaces.Dict(
            {
                'buyContigent': gym.spaces.MultiDiscrete([self.grid_size] * self.edgeServerCount),
                'steerClient': gym.spaces.MultiDiscrete([self.edgeServerCount] * self.clientCount),
            }
        )

    def _get_obs(self):
        return {"clientsLocations": self.clientsLocations, "edgeServerLocations": self.edgeServerLocations
                , "edgeServerPrices": self.edgeServerPrices, "time": self.time
                , "money": self.money}

    def _get_info(self):
        return 'Foo Bar Info'

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Reset variables
        self.time = 0
        self.money = 100_000

        # Reset clients
        self.clientsLocations = self.np_random.integers(0, self.grid_size, size=self.clientCount, dtype=int)
        for c in range(self.clientCount):
            self.clients.append(Client(self.clientsLocations[c]))

        # Reset edge servers
        self.edgeServerLocations = self.np_random.integers(0, self.grid_size, size=self.edgeServerCount, dtype=int)
        self.edgeServerPrices = self.np_random.uniform(0, 10, size=self.edgeServerCount)
        for e in range(self.edgeServerCount):
            self.edgeServers.append(EdgeServer(self.edgeServerLocations[e], self.edgeServerPrices[e]))

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        reward = 0

        # An episode is done when the CP is out of money or time is over
        terminated = self.money <= 0 or self.time >= 100_000

        if not terminated:
            # Buy contigent
            buyContigent = action['buyContigent']
            for index, edgeServer in enumerate(self.edgeServers):
                self.money = edgeServer.sellContigent(buyContigent[index])
            
            # Steer clients
            steerClient = action['steerClient']
            for index, client in enumerate(self.clients):
                client.edgeServer = self.edgeServers[steerClient[index]]
                self.money += client.fetchContent()
                reward += client.fetchContent()

        observation = self._get_obs()
        info = self._get_info()

        self.time += 1

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

    def sellContigent(self, amount):
        self.soldContigent += amount
        return amount * self.price

if __name__ == "__main__":
    env = Sabre(render_mode="human")
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
