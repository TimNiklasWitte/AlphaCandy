import tensorflow as tf
import numpy as np

import sys
sys.path.append("../")
sys.path.append("../AlphaCandy")
sys.path.append("../..")
from CandyCrushGym import *

from DecisionTransformer import *


def main():
    episode_len = 10

    num_candys = 4
    field_size = 8

   
    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
    decisionTransformer.load_weights(f"./saved_models/trained_weights_{field_size}_{num_candys}").expect_partial()

    NUM_RUNS = 1
    for run_idx in range(NUM_RUNS):
                    
        seed = np.random.randint(0, 500000)
        env = CandyCrushGym(seed, field_size=field_size, num_elements=num_candys)

        state = env.reset()

        buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
        buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
        buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

        for episode_idx in range(episode_len):

            buff_states[0, episode_idx, :, :] = state
            buff_rewards[0, episode_idx] = 0.25

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

            # invalid action -> choose valid action
            if not env.isValidAction(best_action):
                best_action = 2

            buff_actions[0, episode_idx] = best_action

            next_state, reward, _, _ = env.step(best_action)
                
            buff_rewards[0, episode_idx] = reward
            state = next_state


            print(reward)

 
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")