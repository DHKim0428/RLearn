import sys
import gym
import pylab
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policies = self.fc_out(x)
        return policies
    

class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = REINFORCE(self.action_size)
        self.optimizer = Adam(lr=self.learning_rate)

        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policies = self.model(state)[0]
        return np.random.choice(self.action_size, 1, p=np.array(policies))[0]

    def record_sample(self, state, action, reward):
        # check dimension of state
        self.states.append(state[0])
        a = np.zeros(self.action_size)
        a[action] = 1
        self.actions.append(a)
        self.rewards.append(reward)

    def discounted_rewards(self):
        n = len(self.rewards)

        discounted_reward = np.zeros_like(self.rewards)
        before = 0
        for t in reversed(range(n)):
            discounted_reward[t] = before * self.discount_factor + self.rewards[t]
            before = discounted_reward[t]
        return discounted_reward

    def train_model(self):
        discounted_rewards = np.float32(self.discounted_rewards())
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(model_params)

            policies = self.model(np.array(self.states))
            actions = np.array(self.actions)
            cross_entropy = -tf.math.log(tf.reduce_sum(actions * policies, axis = 1) + 1e-5)
            loss = tf.reduce_sum(cross_entropy * discounted_rewards)
            self.entropy = np.mean(-policies * tf.math.log(policies))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        self.states, self.actions, self.rewards = [], [], []
    

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []
    score_avg = 0

    num_episode = 3000

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
        
            score += reward
            reward = 0.1 if not done or score == 500 else -1
            
            agent.record_sample(state, action, reward)

            state = next_state

            if done:
                agent.train_model()

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score: {:3.2f} | entropy: {:.3f}".format(
                      e, score, agent.entropy))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

                # 이동 평균이 400 이상일 때 종료
                if score_avg > 400:
                    agent.model.save_weights("./save_model/model", save_format="tf")
                    sys.exit()


    env.close()
            