import os
import datetime

import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
import copy

from Actor import *
from Critic import *

#import sys
#sys.path.append("../AlphaCandy")
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CandyCrushGym import *

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name,size,num):
      
        # Initialization
        # Environment and PPO parameters
        self.field_size = num
        self.num_elements = size

        self.env_name = env_name
        self.seed = np.random.randint(16000)
        self.env = CandyCrushGym(self.seed,self.field_size, self.num_elements)
        self.state_size = self.field_size**2 # Depends on variable state space
        self.action_size = self.env.action_space.n # Depends on variable candy amount
        self.EPISODES = 20 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 50 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 20 # training epochs
        self.shuffle=False
        self.Training_batch = 10000
        self.optimizer = Adam

        self.iterations = 50000

        self.replay_count = 0

        # Logging
        self.writer = SummaryWriter(comment="_"+str(self.field_size)+"_"+str(self.num_elements)+"_"+"PPO")
        
        # Instantiate tb memory
        self.scores_, self.episodes_, self.average_ = [], [], []

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)

        self.Actor_name = f"{self.field_size}_{self.num_elements}_PPO_Actor.h5"
        self.Critic_name = f"{self.field_size}_{self.num_elements}_PPO_Critic.h5"
        
    def act(self, state):

        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
    
    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)

        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)

    def PlotModel(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))

        # saving best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # # decrease learning rate every saved model
            # self.lr *= 0.95
            # K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            # K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""


        return self.average_[-1], SAVING
    
    def run(self): # train only when episode is finished
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done, score, SAVING = False, 0, ''

        i = 0

        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            while not done:

                # Repeat until actor picks a valid action
                while True:
                  action, action_onehot, prediction = self.act(state)

                  if self.env.isValidAction(action):
                    break

                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)

                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size]))
                actions.append(action_onehot)
                rewards.append(reward)
                #dones.append(done)
                predictions.append(prediction)

                # Update current state
                state = np.reshape(next_state, [1, self.state_size])
                score += reward

                i += 1

                if i >= self.iterations:
                  done = True
                
                dones.append(done)

                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))

                    
                    self.writer.add_scalar(f"Var/Score_{self.field_size}_{self.num_elements}_PPO", score, self.episode)
                    self.writer.add_scalar(f"Var/AvgReward_{self.field_size}_{self.num_elements}_PPO", average, self.episode)
                    
                    self.replay(states, actions, rewards, predictions, dones, next_states)

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size])

                    i = 0

                    break

            if self.episode >= self.EPISODES:
                break

        self.env.close()

    def run_batch(self): # train every self.Training_batch episodes
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        done, score, SAVING = False, 0, ''
        i = 0

        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.Training_batch):

                # Actor picks an action
                while True:
                    action, action_onehot, prediction = self.act(state)

                    if self.env.isValidAction(action):
                        break

                # Retrieve new state, reward, and whether the state is terminal
                next_state, reward, done, _ = self.env.step(action)

                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size]))
                actions.append(action_onehot)
                rewards.append(reward)
                #dones.append(done)
                predictions.append(prediction)

                # Update current state
                state = np.reshape(next_state, [1, self.state_size])
                score += reward

                i += 1

                if i >= self.iterations:
                  done = True
                
                dones.append(done)

                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))

                    self.writer.add_scalar(f"Var/Score_{self.field_size}_{self.num_elements}_PPO", score, self.episode)
                    self.writer.add_scalar(f"Var/AvgReward_{self.field_size}_{self.num_elements}_PPO", average, self.episode)

                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size])

                    i = 0
                    
            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= self.EPISODES:
                break
        self.env.close() 


