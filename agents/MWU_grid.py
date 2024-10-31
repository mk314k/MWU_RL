import numpy as np
from .base import RL_Agent, RL_ENV

class MWUAgentGrid(RL_Agent):
    def __init__(self, env: RL_ENV, init_state=None, update_step=1, eta=0.1, gamma=0.9):
        super().__init__(env, init_state, eta)
        self.update_step = update_step
        self.policy_weights = np.ones((self.num_states, self.num_actions))
        self.gamma = gamma
        self.eta = eta
        self.state_visits = np.zeros(self.num_states)
        # Initialize cumulative reward trackers for each state
        self.Rmin_cumm = np.full(self.num_states, np.inf)
        self.Rmax_cumm = np.full(self.num_states, -np.inf)
        self.rewards_table = [[] for _ in range(self.num_states)]

    def choose_action(self, state_idx):
        probs = self.policy_weights[state_idx]
        probs = probs / probs.sum()
        return np.random.choice(self.num_actions, p=probs)

    def update_weights(self, state_idx, action_idx, next_state_idx, reward, done):
        # Update the rewards table for the current state
        self.rewards_table[state_idx].append(reward)
        # Keep only the last 'update_step' rewards
        if len(self.rewards_table[state_idx]) > self.update_step:
            self.rewards_table[state_idx].pop(0)
        # Calculate cumulative reward
        rcumm = sum(self.rewards_table[state_idx])
        # Update Rmin and Rmax for the current state
        self.Rmin_cumm[state_idx] = min(self.Rmin_cumm[state_idx], rcumm)
        self.Rmax_cumm[state_idx] = max(self.Rmax_cumm[state_idx], rcumm)
        # Calculate loss with max-min scaling
        denom = self.Rmax_cumm[state_idx] - self.Rmin_cumm[state_idx]
        if denom == 0:
            loss = 0  # Avoid division by zero
        else:
            loss = 1 - 2 * (rcumm - self.Rmin_cumm[state_idx]) / denom
            loss = np.clip(loss, -1, 1)
        # Update weights multiplicatively
        for a in range(self.num_actions):
            if a == action_idx:
                # Believer
                self.policy_weights[state_idx, a] *= (1 - self.eta * loss)
            else:
                # Non-believer
                self.policy_weights[state_idx, a] *= (1 + self.eta * loss)
        # Normalize weights
        total_weight = self.policy_weights[state_idx].sum()
        if total_weight > 0:
            self.policy_weights[state_idx] /= total_weight
        else:
            # Reinitialize weights if they become zero or negative
            self.policy_weights[state_idx] = np.ones(self.num_actions) / self.num_actions
