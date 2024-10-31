import numpy as np
import random
from .base import RL_Agent


class SARSAgent(RL_Agent):
    def __init__(self, env, init_state=None, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, init_state, eta=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.num_states, self.num_actions))
    
    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.randint(self.num_actions)
        return self.Q[state_idx].argmax()
    
    def update_weights(self, *args):
        state_idx, action_idx, next_state_idx, reward, done = args
        next_action_idx = self.choose_action(next_state_idx)
        current_q = self.Q[state_idx][action_idx]
        next_q = self.Q[next_state_idx][next_action_idx]
        td_error = reward + self.gamma * next_q - current_q
        self.Q[state_idx][action_idx] += self.eta * td_error

class ExpectedSARSAgent(RL_Agent):
    def __init__(self, env, init_state=None, alpha=0.1, gamma=0.9, epsilon=0.1):
        super().__init__(env, init_state, eta=alpha)
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((self.num_states, self.num_actions))
    
    def get_action_probabilities(self, state_idx):
        action_probs = np.ones(self.num_actions) * self.epsilon / self.num_actions
        best_action = self.Q[state_idx].argmax()
        action_probs[best_action] += 1.0 - self.epsilon
        return action_probs
    
    def choose_action(self, state_idx):
        action_probs = self.get_action_probabilities(state_idx)
        return np.random.choice(self.num_actions, p=action_probs)
    
    def update_weights(self, *args):
        state_idx, action_idx, next_state_idx, reward, done = args
        action_probs = self.get_action_probabilities(next_state_idx)
        expected_q = np.dot(self.Q[next_state_idx], action_probs)
        td_error = reward + self.gamma * expected_q - self.Q[state_idx][action_idx]
        self.Q[state_idx][action_idx] += self.eta * td_error
