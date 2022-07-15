from Window import *

import sys
sys.path.append("../")
from CandyCrushGym import *

import time 
def main():

    env = CandyCrushGym()
    env.reset()
    win = Window(env)
    
    
   
    time.sleep(1)

    for i in range(100):

        win.update_game_field()
        
        while True:
            action = np.random.randint(0, 255)
            print("try: ", action)
            reward = 0
            if env.isValidAction(action):
                state, reward, columns_to_fill = env.step_display(action)

            if reward != 0:
                break
        
        print(reward)
        win.update_plots(reward)

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
