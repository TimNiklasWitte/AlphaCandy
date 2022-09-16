import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import sys
sys.path.append("../")
sys.path.append("../AlphaCandy")
sys.path.append("../..")
from CandyCrushGym import *

from DecisionTransformer import *


def main():
    episode_len = 10

    desired_rewards = [0.25, 0.5, 0.75, 1.0, 1.25, 1.50]

    NUM_ROWS = 3
    NUM_RUNS = 100

    fig, ax = plt.subplots(nrows=NUM_ROWS, ncols=1)

    for num_candys in range(4, 7):
        
        plot_idx = num_candys - 4


        rewards = []

        for desired_reward in desired_rewards:

            rewards_desired_reward = []
            for field_size in range(5, 9):

                decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
                decisionTransformer.load_weights(f"../saved_models/trained_weights_{field_size}_{num_candys}").expect_partial()

                avg_reward = 0
                
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

                        avg_reward += np.sum(buff_rewards)

                avg_reward = avg_reward / (NUM_RUNS * episode_len)
                
                rewards_desired_reward.append(avg_reward)

                print(f"{num_candys} {field_size} {avg_reward}")

            rewards.append(rewards_desired_reward)
        y_pos = np.arange(5, 9)

        width=0.8
        num_desired_rewards = len(desired_rewards)
    
        for idx, desired_reward in enumerate(desired_rewards): 
            error = [abs(desired_reward - x) for x in rewards[idx]]
            ax[plot_idx].bar(y_pos - width/2. + idx/float(num_desired_rewards)*width, error, 
                width=width/float(num_desired_rewards), align="edge", label=desired_reward)  

        #ax[plot_idx].bar(y_pos, rewards[0])
        
        print(rewards)
        print(y_pos)
        print("-----------")
        ax[plot_idx].set_title(f"Number of candys: {num_candys}")
        ax[plot_idx].set_xlabel("Field size")
        ax[plot_idx].set_ylabel("| Avg reward\n- desired reward |")

        ax[plot_idx].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax[plot_idx].grid(True)
    

    ax[NUM_ROWS - 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.65),
        fancybox=True, shadow=True, ncol=3, title="Desired rewards")


    fig.set_size_inches(w=6, h=6)
            
    plt.tight_layout()
    plt.savefig("./ErrorDesiredRewardPerformance.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")