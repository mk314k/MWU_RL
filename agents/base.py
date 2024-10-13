from environments.base import RL_ENV

class RL_Agent:
    def __init__(self, env:RL_ENV, init_state, eta=0.1):
        self.env = env
        self.num_states = len(env.S)
        self.num_actions = len(env.A)
        self.eta = eta  # Learning rate
        self.init_state = init_state

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

    def train(self, init_state = None, episodes=1000, record_policy = False, r_thresh=-300):
        episode_rewards = []
        policy_history = []
        if init_state is not None:
            self.init_state = init_state
        for _ in range(0, episodes):
            cum_reward = 0
            state_idx = self.env.state2idx(self.init_state)
            done = False
            while (not done) and cum_reward > r_thresh:
                self.store_state(state_idx)
                action_idx = self.choose_action(state_idx)
                new_state, reward, done = self.env.step(self.env.S[state_idx], self.env.A[action_idx])
                new_state = self.env.state2idx(new_state)
                self.update_weights(state_idx, action_idx, new_state, reward, done)
                state_idx = new_state
                cum_reward += reward
            if record_policy : policy_history.append(list(self.opt_policy().values()))
            episode_rewards.append(reward)
        return episode_rewards, policy_history