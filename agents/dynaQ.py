import random
import numpy as np
from .base import RL_Agent

class DynaQAgent(RL_Agent):
    def __init__(self, env, init_state=None, alpha=0.1, gamma=0.9, epsilon=0.1, planning_steps=5):
        super().__init__(env, init_state, eta=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.model = {}  # For storing transitions
    
    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        return self.Q[state_idx].argmax()
    
    def update_weights(self, *args):
        state_idx, action_idx, next_state_idx, reward, done = args
        # Update Q-values
        max_future_q = self.Q[next_state_idx].max()
        current_q = self.Q[state_idx][action_idx]
        td_error = reward + self.gamma * max_future_q - current_q
        self.Q[state_idx][action_idx] += self.eta * td_error

        # Update model
        self.model[(state_idx, action_idx)] = (reward, next_state_idx)

        # Planning
        for _ in range(self.planning_steps):
            (s, a) = random.choice(list(self.model.keys()))
            r, s_prime = self.model[(s, a)]
            max_future_q = self.Q[s_prime].max()
            self.Q[s][a] += self.eta * (r + self.gamma * max_future_q - self.Q[s][a])
