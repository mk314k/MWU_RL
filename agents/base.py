from environments.base import RL_ENV
import numpy as np

class RL_Agent:
    def __init__(self, env:RL_ENV, init_state=None, eta=0.1):
        self.env = env
        self.num_states = len(env.S)
        self.num_actions = len(env.A)
        self.eta = eta  # Learning rate
        self.state = init_state
        if init_state is None:
            self.state = np.random.randint(self.num_states)

    def step(self):
        action = self.choose_action(self.state)
        next_state, reward, done = self.env.step(self.env.S[self.state], self.env.A[action])
        print(f'Agent at state {self.env.S[self.state]} took action {self.env.A[action]} transition to state {next_state} collecting reward {reward} and is it done? {done}')
        self.state = self.env.state2idx(next_state)
        
    def choose_action(self, state_idx)->int:
        raise NotImplementedError
    
    def __getitem__(self, state_idx):
        return self.choose_action(state_idx)
    
    def get(self, state, alt=None):
        return self.env.A[self.choose_action(self.env.state2idx(state))]
    
    def opt_policy(self):
        return {
            state : self.get(state)
            for state in self.env.S
        }

    def update_weights(self, *args):
        raise NotImplementedError
    
    def store_state(self, state_idx):
        return None

    def train(self, episodes=1000, record_policy = False, evaluate_policy = False ,max_traj = 10):
        episode_rewards = []
        policy_history = []
        policy_eval = []
        for _ in range(0, episodes):
            state_idx = np.random.randint(self.num_states)
            cum_reward = 0
            done = False
            traj = 0
            while (not done) and traj < max_traj:
                self.store_state(state_idx)
                action_idx = self.choose_action(state_idx)
                new_state, reward, done = self.env.step(self.env.S[state_idx], self.env.A[action_idx])
                new_state = self.env.state2idx(new_state)
                self.update_weights(state_idx, action_idx, new_state, reward, done)
                state_idx = new_state
                cum_reward += reward
                traj += 1
            if record_policy : policy_history.append(list(self.opt_policy().values()))
            if evaluate_policy : policy_eval.append(self.env.evaluate_policy(self)[0])
            episode_rewards.append(cum_reward)
        return episode_rewards, policy_history, policy_eval