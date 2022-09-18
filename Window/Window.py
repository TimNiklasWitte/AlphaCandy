from threading import Thread, Event
import tkinter as tk

from Display import *


class Window:
    
    def __init__(self, env, show_plots):
        
        self.event = Event()
        

        self.env = env
        self.show_plots = show_plots

        self.windowThread = Thread(target=self.window_loop)
        self.windowThread.start()

        self.event.wait()        

    

    def window_loop(self):

        self.root = tk.Tk()
        
        self.root.title("Decision Transformer plays Candy Crush")
    
        self.display = Display(master=self.root, env=self.env, show_plots=self.show_plots)

        self.event.set()

        self.display.mainloop()

        
    def update_game_field(self):
        self.display.update_game_field()
     

    def update_plots(self, reward, action_probs, desired_reward, update_stats=True):
        self.display.update_plots(reward, action_probs, desired_reward, update_stats)