import numpy as np
from .base import RL_Agent, RL_ENV

class MWUAgentGrid(RL_Agent):
    def __init__(self, env:RL_ENV, init_state, update_step = 1, eta=0.1, gamma = 0.9):
        super().__init__(env, init_state, eta)
        self.update_step = update_step
        self.policy_weights = np.ones((self.num_states, self.num_actions)) / self.num_actions
        self.r_table = np.zeros(self.update_step)
        self.step = 0
        self.rmin = 0
        self.rmax = 1
        self.gamma = gamma

    def choose_action(self, state_idx):
        probs = self.policy_weights[state_idx]
        return np.random.choice(self.num_actions, p=probs)
    
    def store_state(self, state_idx):
        self.step = self.step + 1

    def update_weights(self, *args):
        state_idx, action_idx, _, reward, _ = args
        idx = self.step % self.update_step
        self.r_table *= self.gamma
        self.r_table[idx] = reward
        if self.step >= self.update_step:
            rcumm = self.r_table.sum()
            self.rmin = min(rcumm, self.rmin)
            self.rmax = max(rcumm, self.rmax)
            regret = np.ones(self.num_actions)
            regret *=  1 - 2 * (rcumm - self.rmin)/(self.rmax - self.rmin + 1e-6)
            regret[action_idx] *= -1
            self.policy_weights[state_idx, :] *= np.exp(self.eta * regret)
            self.policy_weights = self.policy_weights / self.policy_weights.sum(axis=1,keepdims=True)

class MWUAgentGridLinear(MWUAgentGrid):
    def update_weights(self, *args):
        state_idx, action_idx, _, reward, _ = args
        idx = self.step % self.update_step
        self.r_table[idx] = reward
        if self.step >= self.update_step:
            rcumm = self.r_table.sum()
            self.rmin = min(rcumm, self.rmin)
            self.rmax = max(rcumm, self.rmax)
            regret =  1 - 2 * (rcumm - self.rmin)/(self.rmax - self.rmin + 1e-6)
            self.policy_weights[state_idx, :] *= (1 + self.eta * regret)
            self.policy_weights[state_idx, action_idx] *= (1 - self.eta * regret) / (1 + self.eta * regret)
            self.policy_weights = self.policy_weights / self.policy_weights.sum(axis=1,keepdims=True)