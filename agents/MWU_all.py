import numpy as np
from .base import RL_Agent, RL_ENV

class MWUAgentBruteFroce(RL_Agent):
    def __init__(self, env:RL_ENV, init_state, update_step = 1, eta=0.1):
        super().__init__(env, init_state, eta)
        self.update_step = update_step
        self.policies = np.indices(
            (self.num_actions, ) * self.num_states
        ).reshape(self.num_states, -1).T
        self.weights = np.ones(len(self.policies)) / len(self.policies)
        self.r_table = np.zeros((self.num_actions, self.update_step))
        self.state_ids = np.zeros(self.update_step, dtype=np.int16)
        self.state_ids[0] = self.env.state2idx(init_state)
        self.step = 0

    def choose_action(self, state_idx):
        policy_idx = np.random.choice(len(self.policies), p=self.weights)
        return self.policies[policy_idx][state_idx]
    
    def store_state(self, state_idx):
        self.state_ids[self.step] = state_idx
        self.r_table[:, self.step] = np.array([self.env.reward(self.env.S[state_idx], a) for a in self.env.A])
        self.step = (self.step + 1) % self.update_step

    def update_weights(self, *args):
        if self.step == 0:
            rmin = self.r_table.min(axis=0)
            rmax = self.r_table.max(axis=0)
            actions_idx = self.policies[:, self.state_ids]
            rewards = self.r_table[actions_idx, np.arange(self.update_step)]
            regret =  1 - 2 * np.mean((rewards - rmin)/(rmax - rmin + 1e-6), axis=1)
            self.weights *= (1 - self.eta * regret)
            self.weights = self.weights / self.weights.sum()