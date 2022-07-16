from Window import *

import sys
sys.path.append("../")
from CandyCrushGym import *

import time 

from tkinter.constants import *
import tkinter as tk

show_arrow_time = 1
show_swap_time = 1
show_empty_time = 1
drop_candy_time = 0.03

def display_execute_action(action, env, window):      

        #
        # Display arrow
        #

        fieldID = action // env.NUM_DIRECTIONS

        direction = action % env.NUM_DIRECTIONS

        x = fieldID // env.FIELD_SIZE
        y = fieldID % env.FIELD_SIZE

        # top
        if direction == 0:
            img = tk.PhotoImage(file="./Images/Arrows/Top.png")
        # right
        elif direction == 1:
            img = tk.PhotoImage(file="./Images/Arrows/Right.png")
        # down
        elif direction == 2:
            img = tk.PhotoImage(file="./Images/Arrows/Down.png")
        # left
        else:
            img = tk.PhotoImage(file="./Images/Arrows/Left.png")

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

            window.update_plots(reward)
            window.update_game_field()

            time.sleep(show_empty_time) # show also undo swap game state

            return 
        
        window.update_game_field()
        window.update_plots(reward)
     
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

    env = CandyCrushGym()
    env.reset()
    window = Window(env)
    
    
    window.update_game_field()

    for i in range(100):

        window.update_game_field()
        
        while True:
            action = np.random.randint(0, 255)
            
            reward = 0
            if env.isValidAction(action):
                break 
        
        display_execute_action(action, env, window)

    
    


  



 


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
