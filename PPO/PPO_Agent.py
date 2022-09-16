import os
import datetime

import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras import *
tf.compat.v1.disable_eager_execution()
from tensorboardX import SummaryWriter
import copy

from Actor import *
from Critic import *

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CandyCrushGym import *

class Agent_PPO: 

    # PPO Main Optimization Algorithm
    def __init__(self, env_name,size,num):
      
        # Initialization
        # Environment and PPO parameters
        self.field_size = size
        self.num_elements = num
        self.env_name = env_name
        self.seed = np.random.randint(16000) # Get seed for CC gym initialization
        self.env = CandyCrushGym(self.seed,self.field_size, self.num_elements)
        self.state_size = self.field_size**2 # Depends on state space size
        self.action_size = self.env.action_space.n 

        # Set PPO parameters
        self.lr = 0.00025
        self.optimizer = Adam
        self.max_episodes = 500 # Total amount of espisodes
        self.episode = 0 # Keep track of current episode
        self.epochs = 10 # Number of Epochs for Training
        self.batches = 50 # Number of batches collected before each training step
        self.max_iterations = 100 # Maximum number of iterations per episode
        self.best_average = 0 # Initialize best average

        # Start TensorboardX Logger
        self.writer = SummaryWriter(comment="_"+str(self.field_size)+"_"+str(self.num_elements)+"_"+"PPO")
        
        # Create Actor & Critic networks
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

        self.Actor_name = f"{self.field_size}_{self.num_elements}_PPO_Actor.h5"
        self.Critic_name = f"{self.field_size}_{self.num_elements}_PPO_Critic.h5"

        # Initialize Replay Memory
        self.scores_, self.episodes_, self.average_ = [], [], []
        
    # GAE implementation
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

        return np.vstack(gaes), np.vstack(target)
    
    # Replay Memory implementation
    def replay_memory(self, states, actions, rewards, predictions, dones, next_states):

        # Reshape format
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Retrieve value predictions from the Critic 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute GAE values
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # Pack everything into y_true and later unpack it
        y_true = np.hstack([advantages, predictions, actions])
        
        # Train step
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=False)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=False)
    
    # Get prediction of the next action from the Actor   
    def get_action(self, state):

        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    # Save network weights
    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)

    # Determines wheter or not to save the current network weights based on average scores
    def determine_save(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))

        # Only save if current average is better than the current best average
        if self.average_[-1] >= self.best_average: 
            self.best_average = self.average_[-1]
            self.save()
            save_text = "Saving Models"
        else:
            save_text = ""

        return self.average_[-1], save_text

    # Main training Loop
    def run(self): 

        # Reset
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done, score, save_text = False, 0, ''
        i = 0

        while True:

            # Initialize memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []

            # collect batch of size self.batches before training
            for t in range(self.batches):

                # collect episodes
                while True:

                    # Let Actor choose valid action
                    while True:
                        action, action_onehot, prediction = self.get_action(state)

                        if self.env.isValidAction(action):
                            break

                    # Get next state, reward, and done signal
                    next_state, reward, done, _ = self.env.step(action)

                    # Add data to memory
                    states.append(state)
                    next_states.append(np.reshape(next_state, [1, self.state_size]))
                    actions.append(action_onehot)
                    rewards.append(reward)
                    predictions.append(prediction)

                    # Update state
                    state = np.reshape(next_state, [1, self.state_size])

                    # Sum up score of current epoch
                    score += reward

                    # Increase iterations counter
                    i += 1

                    # Check if iteration cap is reached
                    if i >= self.max_iterations:
                      done = True
                    
                    # Check if done is reached
                    dones.append(done)

                    if done:

                        # Increase episode counter
                        self.episode += 1

                        # Get current average and potentially save model
                        average, save_text = self.determine_save(score, self.episode)

                        # Print information
                        print(f"episode: {self.episode}/{self.max_episodes}, score: {score}, average: {round(average,2)} {save_text}")

                        # Add Variable info to TB
                        self.writer.add_scalar(f"Var/Score_{self.field_size}_{self.num_elements}_PPO", score, self.episode)
                        self.writer.add_scalar(f"Var/AvgReward_{self.field_size}_{self.num_elements}_PPO", average, self.episode)

                        # Reset
                        state, done, score, save_text = self.env.reset(), False, 0, ''
                        state = np.reshape(state, [1, self.state_size])
                        i = 0

                        break
                
            # After self.batches episodes update both networks
            self.replay_memory(states, actions, rewards, predictions, dones, next_states)

            # Check if maximum number of episodes is reached
            if self.episode >= self.max_episodes:
                break

        self.env.close()