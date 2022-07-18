import tkinter as tk
from tkinter.constants import *

import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import numpy as np
import sys

class Display(tk.Frame):

    def __init__(self, master, env):
        # Window contains also plots

        self.image_size = 60
        self.window_height = env.FIELD_SIZE * self.image_size + 250
        self.window_width = env.FIELD_SIZE * self.image_size + 1000

        self.root_img_path = sys.path[0]+"/Images"
        self.env = env

        # Game window
        tk.Frame.__init__(self, master)
        master.geometry(f"{self.window_width}x{self.window_height}")     
        self.canvas = tk.Canvas(width=env.FIELD_SIZE * self.image_size, height=env.FIELD_SIZE * self.image_size, bg='black')
        self.canvas.pack(side=LEFT)

        # Plots
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_plot.get_tk_widget().pack(side=RIGHT, fill=tk.BOTH, expand=True)


        self.step_cnt = 0
        self.collected_rewards = []
        self.steps = []

        self.previous_state = np.zeros_like(env.state)
        self.images = []

        self.candies = []

    def convert_normalCandyID_name(self, candyID):
        if candyID == 1:
            return "Red"
        elif candyID == 2:
            return "Orange"
        elif candyID == 3:
            return "Yellow"
        elif candyID == 4:
            return "Green"
        elif candyID == 5:
            return "Blue"
        elif candyID == 6:
            return "Purple"

    def update_game_field(self):
        
        
        for y in range(self.env.FIELD_SIZE):
            for x in range(self.env.FIELD_SIZE):
                candyID = self.env.state[y,x]

                if candyID == -1:
                    self.images[y*self.env.FIELD_SIZE + x] = None
                    self.previous_state[y,x] = -1
                    continue

                if self.previous_state[y,x] == candyID:
                    continue
                
                if self.env.isNormalCandy(candyID):
                    file_name = self.convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Normal/{file_name}.png")
                
                elif self.env.isWrappedCandyID(candyID):
                    candyID = self.env.convertWrappedCandy_toNormal(candyID)
                    file_name = self.convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Wrapped/{file_name}.png") 
                
                elif self.env.isHorizontalStrippedCandy(candyID):
                    candyID = self.env.convertHorizontalStrippedCandy_toNormal(candyID)
                    file_name = self.convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Striped/Horizontal/{file_name}.png")

                elif self.env.isVerticalStrippedCandy(candyID):
                    candyID = self.env.convertVerticalStrippedCandy_toNormal(candyID)
                    file_name = self.convert_normalCandyID_name(candyID)
                    image = tk.PhotoImage(file=f"{self.root_img_path}/Striped/Vertical/{file_name}.png")

                elif candyID == self.env.COLOR_BOMB_CANDY_ID:
                    image = tk.PhotoImage(file=f"{self.root_img_path}/ColourBomb/ColourBomb.png")
                
                if self.previous_state[y,x] == 0:
                    self.images.append(image)
                else:
                    self.images[y*self.env.FIELD_SIZE + x] = image

                candy = self.canvas.create_image(x*self.image_size, y*self.image_size, image=image, anchor=NW)
                self.candies.append(candy)

                self.previous_state[y,x] = candyID


    def update_plots(self, reward):


        noise = np.random.random(256)

        # softmax
        e_x = np.exp(noise - np.max(noise))
        probs = e_x / e_x.sum(axis=0)

        action_top = np.reshape(probs[0::4], newshape=(8,8))
        action_right = np.reshape(probs[1::4], newshape=(8,8))
        action_down = np.reshape(probs[2::4], newshape=(8,8))
        action_left = np.reshape(probs[3::4], newshape=(8,8))

        self.fig.clf()

        #
        # Action: Top
        #
        prob_top_plt = self.fig.add_subplot(331)
        prob_top_plt.set_title("Action: Top")
        img = prob_top_plt.imshow(action_top, cmap='RdBu')
        self.fig.colorbar(img)

        #
        # Action: Right
        #
        prob_right_plt = self.fig.add_subplot(332)
        prob_right_plt.set_title("Action: Right")
        img = prob_right_plt.imshow(action_right, cmap='RdBu')
        self.fig.colorbar(img)

        #
        # State value
        #
        state_value_plt = self.fig.add_subplot(333)
        state_value_plt.set_title("State value")

        value = np.random.random()
        state_value_plt.bar([1], [value], align='center')
        state_value_plt.axes.get_xaxis().set_visible(False)
        state_value_plt.grid(True)

        #
        # Action: down
        #
        prob_down_plt = self.fig.add_subplot(334)
        prob_down_plt.set_title("Action: Down")
        img = prob_down_plt.imshow(action_down, cmap='RdBu')
        self.fig.colorbar(img)

        #
        # Action: left
        #
        prob_left_plt = self.fig.add_subplot(335)
        prob_left_plt.set_title("Action: Left")
        img = prob_left_plt.imshow(action_left, cmap='RdBu')
        self.fig.colorbar(img)

        #
        # Received rewards
        #
        collected_rewards_plt = self.fig.add_subplot(336)

        self.steps.append(self.step_cnt)
        self.collected_rewards.append(reward)

        collected_rewards_plt.plot(self.steps, self.collected_rewards, label="Reward")
        collected_rewards_plt.set_xlim(left=max(0, self.step_cnt - 50), right=self.step_cnt + 50)
        collected_rewards_plt.set_title("Obtained rewards")
        collected_rewards_plt.set_xlabel("Step")
        collected_rewards_plt.set_ylabel("Reward")
        collected_rewards_plt.grid(True)
        collected_rewards_plt.set_ylim(0, 1)
        self.step_cnt += 1

        # Plot mean of collected rewards (not all! only of displayed)
        collected_rewards_part = self.collected_rewards[max(0, self.step_cnt - 50):self.step_cnt]
        mean_collected_rewards_part = np.mean(collected_rewards_part)
        collected_rewards_plt.axhline(mean_collected_rewards_part, color='r', linestyle="--", label="Mean")
        collected_rewards_plt.legend(loc='lower right')

        self.fig.tight_layout()
        self.canvas_plot.draw()