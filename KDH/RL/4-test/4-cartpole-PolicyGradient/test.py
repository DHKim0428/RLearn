import sys
import gym
import copy
import pylab
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
class REINFORCE(tf.keras.Model):
    def __init__(self, action_size):
        super(REINFORCE, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy = self.fc_out(x)
        return policy


# 그리드월드 예제에서의 REINFORCE 에이전트
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        self.model = REINFORCE(self.action_size)
        self.model.load_weights('save_model/model')

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        policies = self.model(state)[0]
        return np.random.choice(self.action_size, 1, p=np.array(policies))[0]


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    
    action_size = env.action_space.n
    agent = REINFORCEAgent(state_size, action_size)

    EPISODES = 10
    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            env.render()

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            score += reward

            state = next_state

            if done:
                print("episode: {:3d} | score: {:3f}".format(e, score))

    env.close()