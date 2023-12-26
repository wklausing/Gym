import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class GymSabreRender:
    
    # Update function for animation
    def update(self, step):
        # Simulate some movement or changes (this is just an example)
        self.client_positions[:, 0] += np.random.rand(self.num_clients) * 0.01 - 0.005
        self.client_positions[:, 1] += np.random.rand(self.num_clients) * 0.01 - 0.005

        # Update data
        self.clients.set_offsets(self.client_positions)
        self.servers.set_offsets(self.server_positions)
        self.time_text.set_text(f'Time: {step}')

        return self.clients, self.servers, self.time_text

    def render(self):
        # Create animation
        FuncAnimation(self.fig, self.update, frames=self.num_steps, interval=500, blit=True)

    def init2(self):
        # Settings
        self.num_clients = 10
        self.num_servers = 5
        self.num_steps = 20  # Total number of time steps

        # Initialize positions
        self.client_positions = np.random.rand(self.num_clients, 2)
        self.server_positions = np.random.rand(self.num_servers, 2)

        # Initialize the plot
        self.fig, self.ax = plt.subplots()
        self.clients = self.ax.scatter([], [], color='blue', label='Clients')
        self.servers = self.ax.scatter([], [], color='red', label='Servers')
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)

        # Additional plot settings
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_title('Dynamic Map of Clients and Servers')
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.legend()
        self.ax.grid(True)
    
        # Show the animation
        plt.show()

    def __init__(self) -> None:
        print('Here')
        self.init2()

if __name__ == "__main__":
    GymSabreRender()
