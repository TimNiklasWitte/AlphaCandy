from Window import *

import sys
sys.path.append("../")
from CandyCrushGym import *

def main():

    env = CandyCrushGym()
    win = Window(env)
    
    win.update_game_field()
    win.update_plots(0)

    for x in range(1000):
        try:
            state, reward = env.step(x)
            print(reward)
            win.update_game_field()
            win.update_plots(reward)

        except:
            pass



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
