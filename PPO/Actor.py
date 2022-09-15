import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K

import numpy as np

class Actor_Model():
    def __init__(self, input_shape, action_space, lr, optimizer):

        # Format input
        input_Act = Input(input_shape)
        self.action_space = action_space

        # Define Network structure
        X_Act = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(input_Act)
        X_Act = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_Act)
        X_Act = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_Act)

        # Get final output
        output_Act = Dense(self.action_space, activation="softmax")(X_Act)

        # Create Actor
        self.Actor = Model(inputs = input_Act, outputs = output_Act)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))

    # Define ProximalPoliyOptimization actor loss
    def ppo_loss(self, y_true, y_pred):

        # Use GAEs
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        # Get probs
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        # Clip
        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        # Calculate update ratio
        ratio = K.exp(K.log(prob) - K.log(old_prob))

        #Calculate actor loss PPO style
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))

        # Calculate entropy
        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        # Get final loss
        total_loss = actor_loss - entropy

        return total_loss

    # Define prediction process
    def predict(self, state):
        return self.Actor.predict(state)