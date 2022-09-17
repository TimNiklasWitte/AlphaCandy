from Window import *

import sys
sys.path.append("../../AlphaCandy")
from CandyCrushGym import *
sys.path.append("../Decision Transformer")
from DecisionTransformer import *


import time 

from tkinter.constants import *
import tkinter as tk

import numpy as np
import tensorflow as tf

show_arrow_time = 1
show_swap_time = 1
show_empty_time = 1
drop_candy_time = 0.03

def display_execute_action(action, action_probs, env, window):      

        #
        # Display arrow
        #

        fieldID = action // env.NUM_DIRECTIONS

        direction = action % env.NUM_DIRECTIONS

        x = fieldID // env.FIELD_SIZE
        y = fieldID % env.FIELD_SIZE

        # top
        if direction == 0:
            img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Top.png")
        # right
        elif direction == 1:
            img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Right.png")
        # down
        elif direction == 2:
            img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Down.png")
        # left
        else:
            img = tk.PhotoImage(file=sys.path[0]+"/Images/Arrows/Left.png")

        # top or down
        if direction == 0 or direction == 2:
            window.display.canvas.create_image(x * window.display.image_size , y * window.display.image_size, image=img, anchor=NW)
                
        # right or left
        else:
            window.display.canvas.create_image(x * window.display.image_size, y * window.display.image_size, image=img, anchor=NW)

        time.sleep(show_arrow_time)
        img = None

        #
        # Swap
        #

        # Swap candy
        x_swap = x # attention: numpy x->y are swapped
        y_swap = y # attention: numpy x->y are swapped
        # top
        if direction == 0:
            y_swap += -1
        # down
        elif direction == 2: 
            y_swap += 1
        # right 
        elif direction == 1:
            x_swap += 1
        # left 
        elif direction == 3:
            x_swap += -1

        # swap
        tmp = env.state[y,x]
        env.state[y,x] = env.state[y_swap, x_swap]
        env.state[y_swap, x_swap] = tmp

        window.update_game_field()
        time.sleep(show_swap_time)

        #
        # React
        #
        reward = env.react(x,y, x_swap, y_swap)
     
        if reward == 0:
            tmp = env.state[y,x]
            env.state[y,x] = env.state[y_swap, x_swap]
            env.state[y_swap, x_swap] = tmp

            window.update_plots(reward, action_probs)
            window.update_game_field()

            time.sleep(show_empty_time) # show also undo swap game state

            return 
        
        window.update_game_field()
        window.update_plots(reward, action_probs)
     
        time.sleep(show_empty_time)

        #
        # Fill 
        #

        columns_to_fill = list(env.columns_to_fill)
        env.columns_to_fill = set()

        while len(columns_to_fill) != 0:
    
            for idx, column_idx in enumerate(columns_to_fill):

                done = True
                for x in range(env.FIELD_SIZE):

                    if env.state[x, column_idx] == -1:
                        
                        done = False
                        if x - 1 < 0:
                            candy = np.random.randint(1, env.NUM_ELEMENTS)
                        else:
                            candy = env.state[x - 1, column_idx]
                            env.state[x - 1, column_idx] = -1
        
                        env.state[x, column_idx] = candy
                        
                        time.sleep(drop_candy_time)

                        window.update_game_field()
                        window.display.previous_state[x, column_idx] = candy
                        

                if done:
                    columns_to_fill.pop(idx)

        window.update_game_field()

def main():

    episode_len = 10
    num_candys = 4
    field_size = 8


    decisionTransformer = DecisionTransformer(episode_len,num_actions=field_size*field_size*4)
    decisionTransformer.load_weights(f"../Decision Transformer/saved_models/trained_weights_{field_size}_{num_candys}").expect_partial()

    seed = np.random.randint(0, 500000)
    env = CandyCrushGym(seed, field_size=field_size, num_elements=num_candys)

    window = Window(env)
    window.update_game_field()

    state = env.reset()

    buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
    buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
    buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

    desired_reward = 0.25

    buff_states[0, 0, :, :] = state
    buff_rewards[0, 0] = desired_reward

    none_action_id = field_size*field_size*4 + 1
    buff_actions[0, 0:episode_len] = none_action_id

    

    while True:
        
        window.update_game_field()

        reward = 0
        episode_idx = 0
        
        cnt_zero = 0
        while reward == 0:

            # buff_states[0, episode_idx, :, :] = state
            # buff_rewards[0, episode_idx] = 0.25

            # none_action_id = field_size*field_size*4 + 1
            # buff_actions[0, episode_idx:episode_len] = none_action_id

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

            # buff_actions[0, episode_idx] = best_action

            next_state, reward, _, _ = env.step(best_action)
                    
            #buff_rewards[0, episode_idx] = reward
            state = next_state

            episode_idx += 1

    
            if episode_idx < episode_len - 1:

                buff_states[0, episode_idx, :, :] = state
                buff_rewards[0, episode_idx] = desired_reward
                buff_actions[0, episode_idx] = best_action


                # buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
                # buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
                # buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

                # buff_states[0, 0, :, :] = state
                # buff_rewards[0, 0] = desired_reward

                # none_action_id = field_size*field_size*4 + 1
                # buff_actions[0, 0:episode_len] = none_action_id

                # episode_idx = 0
           
                # state = env.reset()
                # window.update_game_field()


            else:
                
                buff_states[0, 0:episode_len-1, :, :] = buff_states[0, 1:episode_len, :, :]
                buff_states[0, -1, :, :] = state

                buff_rewards[0, 0:episode_len-1] = buff_rewards[0, 1:episode_len]
                buff_rewards[0, -1] = 0.25

                buff_actions[0, 0:episode_len-1] = buff_actions[0, 1:episode_len]
                buff_actions[0, -1] = best_action


            if reward == 0:
                cnt_zero += 1
            else:
                cnt_zero = 0
            
            if cnt_zero == episode_len:


                buff_states = np.zeros(shape=(1, episode_len, field_size, field_size), dtype=np.uint8)
                buff_actions = np.zeros(shape=(1, episode_len,), dtype=np.uint8)
                buff_rewards = np.zeros(shape=(1, episode_len,), dtype=np.float32)

                buff_states[0, 0, :, :] = state
                buff_rewards[0, 0] = desired_reward

                none_action_id = field_size*field_size*4 + 1
                buff_actions[0, 0:episode_len] = none_action_id

                state = env.reset()
                window.update_game_field()
                episode_idx = 0

                cnt_zero = 0
                print("reset")

        display_execute_action(best_action, action[:-1], env, window)
        print(reward)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
