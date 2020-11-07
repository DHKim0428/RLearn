import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DeepSARSA(tf.keras.Model):
    def __init__(self, action_size):
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


class DeepSARSAAgent:
    def __init__(self, state_size, action_size):
        self.render = True

        self.state_size = state_size
        self.action_size = action_size
        
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.model = DeepSARSA(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(action_size)
        else:
            q_values = self.model(state)[0]
            return np.argmax(q_values)

    def train_model(self, state, action, reward, next_state, next_action, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            
        