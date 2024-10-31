class RL_ENV:
    def __init__(self, state_space, action_space, terminal_states = None) -> None:
        self.S = state_space
        self.A = action_space
        self.T = set() if terminal_states is None else terminal_states
        self.IAS = self.T #inactive states
    
    def state2idx(self, state)->int:
        raise NotImplementedError
    
    def action2idx(self, action)->int:
        raise NotImplementedError
    
    def transition_state(self, state, action):
        raise NotImplementedError
        
    def reward(self, state, action, next_state=None):
        raise NotImplementedError
    
    def step(self, state, action):
        if state in self.T:
            return state, 0, True
        next_state = self.transition_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state in self.T
        return next_state, reward, done
    
    def plot(self, agent_state=None, policy=None):
        raise NotImplementedError
    
    def evaluate_policy(self, policy, max_step=200, plot=False):
        report = {}
        ts = 0
        for s in self.S:
            if s in self.IAS:
                continue
            done = False
            x = s
            score = 0
            step = 0
            while (not done) and step < max_step:
                x, r, done = self.step(x, policy.get(x))
                score += r
                step += 1
            report[s] = score
            ts += score
        if plot:
            self.plot(policy=policy)
        return ts, report