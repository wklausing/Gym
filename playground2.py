import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class DynamicMap:
    def __init__(self, num_clients, num_servers, num_steps):
        self.num_clients = num_clients
        self.num_servers = num_servers
        self.num_steps = num_steps

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.clients = self.ax.scatter([], [], color='blue', label='Clients')
        self.servers = self.ax.scatter([], [], color='red', label='Servers')
        self.lines = [self.ax.plot([], [], 'k-', lw=0.5)[0] for _ in range(num_clients)]
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

        # Additional plot settings
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Dynamic Map of Clients and Servers')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.legend()
        self.ax.grid(True)

    def get_new_data(self, step):
        # Placeholder for fetching new data
        new_client_positions = np.random.rand(self.num_clients, 2)
        new_server_positions = np.random.rand(self.num_servers, 2)

        return new_client_positions, new_server_positions

    def update(self, step):
        # Fetch new data for each step
        client_positions, server_positions = self.get_new_data(step)

        # Update client and server positions
        self.clients.set_offsets(client_positions)
        self.servers.set_offsets(server_positions)

        # Update lines from each client to a server
        for line, client_pos in zip(self.lines, client_positions):
            server_index = np.random.choice(self.num_servers)  # Randomly choose a server for each client
            server_pos = server_positions[server_index]
            line.set_data([client_pos[0], server_pos[0]], [client_pos[1], server_pos[1]])

        self.time_text.set_text(f'Time: {step}')

        return [self.clients, self.servers, *self.lines, self.time_text]

    def animate(self):
        # Create animation
        ani = FuncAnimation(self.fig, self.update, frames=self.num_steps, interval=500, blit=True)
        plt.show()

# Usage
dynamic_map = DynamicMap(num_clients=10, num_servers=5, num_steps=20)
dynamic_map.animate()
