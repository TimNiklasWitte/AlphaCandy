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

    # arrow_top_img = tk.PhotoImage(file="./Images/Arrows/Top.png")
    # arrow_down_img = tk.PhotoImage(file="./Images/Arrows/Down.png")
    # arrow_right_img = tk.PhotoImage(file="./Images/Arrows/right.png")
    # arrow_left_img = tk.PhotoImage(file="./Images/Arrows/right.png")

    # win.display.canvas.create_image(60, 90, image=image, anchor=NW)
    # time.sleep(10)

    for i in range(100):

        win.update_game_field()
        
        while True:
            action = np.random.randint(0, 255)
            print("try: ", action)
            reward = 0
            if env.isValidAction(action):

                
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
                # down
                elif direction == 1:
                    img = tk.PhotoImage(file="./Images/Arrows/Down.png")
                # right
                elif direction == 2:
                    img = tk.PhotoImage(file="./Images/Arrows/Right.png")
                # left
                else:
                    img = tk.PhotoImage(file="./Images/Arrows/Left.png")

                # top or down
                if direction == 0 or direction == 1:
                    win.display.canvas.create_image(x * win.display.image_size , y * win.display.image_size + win.display.image_size//2, image=img, anchor=NW)
                
                # right or left
                else:
                    win.display.canvas.create_image(x * win.display.image_size + win.display.image_size//2, y * win.display.image_size, image=img, anchor=NW)

                time.sleep(1.5)
                img = None

                state, reward, columns_to_fill = env.step_display(action)

                win.update_game_field()
                win.update_plots(reward)
                print(reward)
                time.sleep(1.5)

                break 
        
        
        

        columns_to_fill = list(columns_to_fill)
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
