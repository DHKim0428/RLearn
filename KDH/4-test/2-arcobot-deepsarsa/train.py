import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
from numpy import cos, sin, pi
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
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)[0]
            return np.argmax(q_values)
    
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            predict = self.model(state)[0][action]

            next_q = reward + (1-done) * self.discount_factor * self.model(next_state)[0][next_action]

            loss = tf.reduce_mean(tf.square(next_q - predict))
        
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    env = gym.make('Acrobot-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DeepSARSAAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 300
    for e in range(num_episode):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])
        action = agent.get_action(state)

        while not done:
            if agent.render:
                env.render()

            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)

            score += reward
            # reward 수정??
            # reward = 1 - 0.1 * (-cos(state[0][0]) - cos(state[0][1] + state[0][0]))

            agent.train_model(state, action, reward, next_state, next_action, done)

            state = next_state
            action = next_action

            if done:
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score: {:3.2f} | epsilon: {:.3f}".format(
                      e, score, agent.epsilon))


                # 에피소드마다 학습 결과 그래프로 저장
                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")
        
    agent.model.save_weights("./save_model/model", save_format = "tf")
    env.close()
