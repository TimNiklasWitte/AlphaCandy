from Window import *

import sys
sys.path.append("../")
from CandyCrushGym import *

import time 

from tkinter.constants import *
import tkinter as tk

def main():

    env = CandyCrushGym()
    env.reset()
    win = Window(env)
    
    
    win.update_game_field()

    for i in range(100):

        win.update_game_field()
        
        while True:
            action = np.random.randint(0, 255)
            
            reward = 0
            if env.isValidAction(action):

                
                #
                # Display arrow
                #

                fieldID = action // env.NUM_DIRECTIONS

                direction = action % env.NUM_DIRECTIONS

                x = fieldID // env.FIELD_SIZE
                y = fieldID % env.FIELD_SIZE

                print(f"x: {x} y: {y}")

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
                    win.display.canvas.create_image(x * win.display.image_size , y * win.display.image_size - win.display.image_size//2, image=img, anchor=NW)
                
                # right or left
                else:
                    win.display.canvas.create_image(x * win.display.image_size + win.display.image_size//2, y * win.display.image_size, image=img, anchor=NW)

                time.sleep(1)
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


                win.update_game_field()
                time.sleep(1)

                reward = env.react(x,y, x_swap, y_swap)
                #state, reward, columns_to_fill = env.step_display(action)

                if reward == 0:
                    tmp = env.state[y,x]
                    env.state[y,x] = env.state[y_swap, x_swap]
                    env.state[y_swap, x_swap] = tmp

                win.update_game_field()
                win.update_plots(reward)
                print(reward)
                time.sleep(1)

                break 
        
        
        

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
        
                        #win.display.canvas.move(win.display.candies[column_idx*env.FIELD_SIZE + x], x * 60, 0)

                        env.state[x, column_idx] = candy
                        

                        time.sleep(0.03)

                        win.update_game_field()
                        win.display.previous_state[x, column_idx] = candy
                        #

                if done:
                    columns_to_fill.pop(idx)

        print(env.state)
        win.update_game_field()
    

    
    # y*self.env.FIELD_SIZE + x

    #
    #win.update_game_field()
    
  



 


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
