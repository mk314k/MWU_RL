import numpy as np
from .base import RL_Agent, RL_ENV

class MWUAgentQGrid(RL_Agent):
    def __init__(self, env: RL_ENV, init_state=None, update_step=1, eta=0.1, gamma=0.9, lambda_=0.8):
        super().__init__(env, init_state, eta)
        self.update_step = update_step
        self.policy_weights = np.ones((self.num_states, self.num_actions))
        self.gamma = gamma
        self.eta = eta
        self.lambda_ = lambda_  # Trace decay parameter
        self.state_visits = np.zeros(self.num_states)
        self.eligibility_traces = np.zeros((self.num_states, self.num_actions))
        self.V = np.zeros(self.num_states)
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.step = 0  # To track time steps for adaptive eta

    def choose_action(self, state_idx):
        probs = self.policy_weights[state_idx]
        probs = probs / probs.sum()
        return np.random.choice(self.num_actions, p=probs)

    def update_eligibility_traces(self, state_idx, action_idx):
        # Decay existing traces
        self.eligibility_traces *= self.gamma * self.lambda_
        # Increment trace for current state-action pair
        self.eligibility_traces[state_idx, action_idx] += 1

    def compute_td_error(self, state_idx, next_state_idx, reward, done):
        V_s_t = self.V[state_idx]
        V_s_tp1 = self.V[next_state_idx] if not done else 0
        delta_t = reward + self.gamma * V_s_tp1 - V_s_t
        return delta_t

    def update_weights(self, state_idx, action_idx, next_state_idx, reward, done):
        # Update learning rate
        self.eta = self.eta / (1 + 1e-5 * self.step)
        self.step += 1

        # Compute TD error
        delta_t = self.compute_td_error(state_idx, next_state_idx, reward, done)
        # Update eligibility traces
        self.update_eligibility_traces(state_idx, action_idx)
        # Update log weights
        loss = -delta_t * self.eligibility_traces
        log_policy_weights = np.log(self.policy_weights + 1e-8)  # Add small constant to prevent log(0)
        log_policy_weights -= self.eta * loss
        self.policy_weights = np.exp(log_policy_weights)
        # Normalize weights for the current state
        self.policy_weights[state_idx] /= self.policy_weights[state_idx].sum()
        # Update value function estimates
        self.V[state_idx] += self.eta * delta_t
        self.Q[state_idx, action_idx] += self.eta * delta_t

class QLearningExpertAgent(RL_Agent):
    def __init__(self, env: RL_ENV, init_state=None, eta=0.1, gamma=0.9):
        super().__init__(env, init_state, eta)
        self.gamma = gamma
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.policy_weights = np.ones((self.num_states, self.num_actions)) / self.num_actions
    
    def choose_action(self, state_idx):
        probs = self.policy_weights[state_idx]
        return np.random.choice(self.num_actions, p=probs)
    
    def update_weights(self, *args):
        state_idx, action_idx, next_state_idx, reward, done = args
        max_future_q = self.Q[next_state_idx].max()
        current_q = self.Q[state_idx][action_idx]
        td_error = reward + self.gamma * max_future_q - current_q
        self.Q[state_idx][action_idx] += self.eta * td_error

        # Update policy weights using MWU based on Q-values
        q_values = self.Q[state_idx]
        self.policy_weights[state_idx, :] *= np.exp(q_values)
        self.policy_weights[state_idx] /= self.policy_weights[state_idx].sum()
