import numpy as np
from matplotlib import pyplot as plt

def softmax(x):
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def train_debug(agent, init_state = None):
    if init_state is not None:
        agent.init_state = init_state
    cum_reward = 0
    state_idx = agent.env.state2idx(agent.init_state)
    done = False
    while (not done) and cum_reward > -300:
        agent.store_state(state_idx)
        print(state_idx, "action_end")
        action_idx = agent.choose_action(state_idx)
        new_state, reward, done = agent.env.step(agent.env.S[state_idx], agent.env.A[action_idx])
        new_state_idx = agent.env.state2idx(new_state)
        print(new_state, new_state_idx)
        agent.update_weights(state_idx, action_idx, new_state_idx, reward, done)
        state_idx = new_state_idx
        cum_reward += reward
    agent.init_state = new_state
    return cum_reward