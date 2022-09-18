from threading import Thread, Event
import tkinter as tk

from Display import *

class Window:
    
    def __init__(self, env):
        
        self.event = Event()

        self.env = env
        self.windowThread = Thread(target=self.window_loop)
        self.windowThread.start()

        self.event.wait()        

    

    def window_loop(self):

        self.root = tk.Tk()
        self.root.title("AlphaCandy")
  
        self.display = Display(master=self.root, env=self.env)

        self.event.set()

        self.display.mainloop()
    

    def update_game_field(self):
        self.display.update_game_field()
    
    def update_plots(self, reward, action_probs):
        self.display.update_plots(reward, action_probs)