import numpy as np

import sys
sys.path.append("../..")
from CandyCrushGym import *

class EnvManager:

    def __init__(self):
        
        self.gym = CandyCrushGym()
    
        self.window_size = 4

        self.reset()

    def step(self, action: int):
        self.step_cnt += 1

        current_state, reward = self.gym.step(action)

        if self.step_cnt < self.window_size:
            self.state[self.step_cnt, ...] = current_state
        else:
            self.state[0:self.window_size - 2, ...] = self.state[1:self.window_size - 1, ...]
            self.state[self.window_size - 1, ...] = current_state
        return self.state, reward

    def gym_step(self, action: int):
        current_state, reward = self.gym.step(action)
        return current_state, reward
    
    def save(self, current_state):
        self.step_cnt += 1

        if self.step_cnt < self.window_size:
            self.state[self.step_cnt, ...] = current_state
        else:
            self.state[0:self.window_size - 2, ...] = self.state[1:self.window_size - 1, ...]
            self.state[self.window_size - 1, ...] = current_state

        return self.state

    def reset(self):

        current_state = self.gym.reset()
        self.state = np.zeros(shape=(self.window_size, *current_state.shape), dtype=np.int8)
        self.state[0, ...] = current_state

        self.step_cnt = 0

        return self.state
    
    def isValidAction(self, action):
        return self.gym.isValidAction(action)

    @property
    def observation_space_shape(self):
        return self.state.shape

    @property
    def action_space_n(self):
        return self.gym.action_space_n
