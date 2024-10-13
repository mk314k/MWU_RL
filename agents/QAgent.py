from .base import RL_Agent
import random
import numpy as np

class QLearningAgent(RL_Agent):
    def __init__(self, env, init_state, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, init_state, eta=alpha)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.init_state = init_state

    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        return self.Q[state_idx].argmax()

    def update_weights(self, *args):
        state_idx, action_idx, next_state_idx, reward, _ = args
        max_future_q = self.Q[next_state_idx].max()
        current_q = self.Q[state_idx][action_idx]
        self.Q[state_idx][action_idx] += self.eta * (reward + self.gamma * max_future_q - current_q)
