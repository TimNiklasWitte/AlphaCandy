import tensorflow as tf
import numpy as np
import sys

sys.path.append("../")
sys.path.append("../AlphaCandy")
from CandyCrushGym import *

from DecisionTransformer import *


def main():
    episode_len = 10
    field_size = 6

    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
    decisionTransformer.load_weights("./saved_models/trained_weights_epoch_2").expect_partial()

    env = CandyCrushGym(0, 6, 6)
    
    step_cnt = 0
    state = env.reset()

    buff_state = np.zeros(shape=(episode_len, field_size, field_size), dtype=np.uint8)
    buff_actions = np.zeros(shape=(episode_len,), dtype=np.uint8)
    buff_rewards = np.zeros(shape=(episode_len,), dtype=np.float32)

   
    for i in range(10):

        buff_state[step_cnt, :, :] = state
        buff_rewards[step_cnt] = 1

        x1 = np.expand_dims(buff_state, axis=0)
        x2 = np.expand_dims(buff_rewards, axis=0)

        action = decisionTransformer(x1, x2)
        action = action[0]
        next_state, reward, _, _ = env.step(action)
        print(reward)
        buff_rewards[step_cnt] = reward
        state = next_state
        step_cnt += 1

    print(state)

    #next_state, reward, _, _ = env.step(action)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")