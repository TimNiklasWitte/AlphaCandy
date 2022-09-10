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
    decisionTransformer.load_weights("./saved_models/trained_weights_epoch_13").expect_partial()

    env = CandyCrushGym(0, 6, 6)
    
    step_cnt = 0
    state = env.reset()

    buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
    buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
    buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

   
    for episode_idx in range(episode_len):

        buff_states[0, step_cnt, :, :] = state
        buff_rewards[0, step_cnt] = 0.5

        none_action_id = field_size*field_size*4 + 1
        buff_actions[0, episode_idx:episode_len] = none_action_id

        #
        # Preprocess
        #

        # onehotify states
        states = np.reshape(buff_states, newshape=(1*episode_len*field_size*field_size))
        num_one_hot = 26 # num of candys
        states = tf.one_hot(states, depth=num_one_hot)
        states = tf.reshape(states, shape=(1, episode_len, field_size, field_size, num_one_hot))

        # onehotify actions
        num_actions = field_size*field_size*4 + 1
        actions = tf.one_hot(buff_actions, depth=num_actions)

        action = decisionTransformer(states, actions, buff_rewards)
        action = action[0] # remove batch dim

        best_action = np.argmax(action)
     
        buff_actions[0, episode_idx] = best_action

        next_state, reward, _, _ = env.step(best_action)
  
        buff_rewards[0, step_cnt] = reward
        state = next_state
        step_cnt += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")