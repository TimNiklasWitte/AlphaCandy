import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K

import numpy as np

class Critic_Model():
    def __init__(self, input_shape, action_space, lr, optimizer):

        # Format input
        input_Crit = Input(input_shape)
        old_values = Input(shape=(1,))

        # Define Network structure
        X_Crit = Dense(512, activation="relu", kernel_initializer='he_uniform')(input_Crit)
        X_Crit = Dense(256, activation="relu", kernel_initializer='he_uniform')(X_Crit)
        X_Crit = Dense(64, activation="relu", kernel_initializer='he_uniform')(X_Crit)
        output_Crit = Dense(1, activation=None)(X_Crit)

        # Create Critic
        self.Critic = Model(inputs=[input_Crit, old_values], outputs = output_Crit)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    # Define ProximalPoliyOptimization2 critic loss
    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):

            # Calculate clipped loss
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            # Use tweaked PPO loss function to get final loss
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))

            return value_loss
        return loss

    # Define prediction process
    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])