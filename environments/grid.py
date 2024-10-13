import numpy as np
from matplotlib import pyplot as plt
from .base import RL_ENV

class GridWorldEnv(RL_ENV):
    def __init__(self, grid_size=(5, 5), goal_state=(4, 4), walls=None, damps=None):
        super().__init__(
            [(x, y) for x in range(grid_size[0]) for y in range(grid_size[1])],
            ['up', 'down', 'left', 'right']
        )
        self.grid_size = grid_size
        self.goal_state = {goal_state}
        self.walls = walls if walls else set()
        self.damps = damps if damps else set()
        self.T = self.goal_state.union(self.walls)
    
    def state2idx(self, state):
        x, y = state
        return x * self.grid_size[1] + y
    
    def action2idx(self, action):
        return self.A.index(action)
    
    def dist_func(self, state):
        td = 0
        for gs in self.goal_state:
            td += (state[0] - gs[0]) ** 2 + (state[1] - gs[1]) ** 2
        return td
        
    def transition_state(self, state, action):
        x, y = state
        if action == 'down':
            y = max(0, y - 1)
        elif action == 'up':
            y = min(self.grid_size[1] - 1, y + 1)
        elif action == 'left':
            x = max(0, x - 1)
        elif action == 'right':
            x = min(self.grid_size[0] - 1, x + 1)
        return (x, y)
    
    def reward(self, state, action, next_state=None):
        if next_state is None:
            next_state = self.transition_state(state, action)
        reward = -5 + 4 * (self.dist_func(state) - self.dist_func(next_state))  # Penalty for each move
        # print(reward)
        if next_state in self.goal_state:
            reward += 120  # Reward for reaching the goal
        elif next_state in self.damps:
            reward -= 60
        elif next_state in self.walls or state == next_state:
            reward -= 100 
        return reward

    
    def plot(self, agent_state=None, policy=None):
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Draw the grid
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if (x, y) in self.walls:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='red'))  # Obstacles
                elif (x, y) in self.damps:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='blue'))  # Obstacles
                else:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, edgecolor='black', fill=False))
                
                if (x, y) in self.goal_state:
                    ax.text(x + 0.5, y + 0.5, 'G', va='center', ha='center', fontsize=12, color='green')  # Goal state

                if agent_state and (x, y) == agent_state:
                    ax.text(x + 0.5, y + 0.5, 'A', va='center', ha='center', fontsize=12, color='blue')  # Agent

        # Plot the policy arrows
        if policy:
            for state, action in policy.items():
                x, y = state
                x_plot = x + 0.5
                y_plot = y + 0.5

                if action == 'up':
                    ax.arrow(x_plot, y_plot, 0, 0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 'down':
                    ax.arrow(x_plot, y_plot, 0, -0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 'left':
                    ax.arrow(x_plot, y_plot, -0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')
                elif action == 'right':
                    ax.arrow(x_plot, y_plot, 0.4, 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

        ax.set_xticks(np.arange(self.grid_size[0]))
        ax.set_yticks(np.arange(self.grid_size[1]))
        ax.set_xlim([0, self.grid_size[0]])
        ax.set_ylim([0, self.grid_size[1]])
        plt.show()
