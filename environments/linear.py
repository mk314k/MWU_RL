from .base import RL_ENV
from matplotlib import pyplot as plt
import numpy as np

class LinearWorldEnv(RL_ENV):
    def __init__(self, grid_size=5, goal_state = 5):
        super().__init__(
            np.arange(1, grid_size + 1, 1),
            [-1, 1],
            {goal_state}
        )
        self.grid_size = grid_size
        self.goal_state = goal_state

    def state2idx(self, state):
        return int(state - 1)
    
    def action2idx(self, action):
        return int(action / 2 + 0.5)
    
    def transition_state(self, state, action):
        next_state = state + action
        if next_state > self.grid_size:
            return self.grid_size
        elif next_state < 1:
            return 1
        return next_state
        
    def reward(self, state, action, next_state=None):
        if next_state is None:
            next_state = self.transition_state(state, action)
        reward =  10 if abs(self.goal_state - next_state) < abs(self.goal_state - state) else -20
        if next_state == self.goal_state:
            reward += 100
        return reward
    
    
    def plot(self, agent_state=None, policy=None):
        # Create a plot to visualize the grid
        fig, ax = plt.subplots(figsize=(10, 1))
        
        # Plot the grid
        for state in self.S:
            ax.add_patch(plt.Rectangle((state-0.5, 0), 1, 1, edgecolor='black', fill=False))
            if state == self.goal_state:
                ax.text(state, 0.5, 'G', va='center', ha='center', fontsize=12, color='green')  # Goal state

            # Agent position
            if agent_state and state == agent_state:
                ax.text(state, 0.5, 'A', va='center', ha='center', fontsize=12, color='blue')  # Agent

            if policy and state not in self.T:
                ax.arrow(state, 0.5, 0.4 * policy.get(state), 0, head_width=0.2, head_length=0.2, fc='black', ec='black')

        ax.set_xlim([0.5, self.grid_size + 0.5])
        ax.set_ylim([0, 1])
        ax.set_xticks(self.S)
        ax.set_yticks([])  # No y-axis ticks needed for linear environment
        plt.show()