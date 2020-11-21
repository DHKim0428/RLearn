import sys
from environment import Env
import pylab
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from tensorflow_probability import distributions as tfd


# 정책 신경망과 가치 신경망 생성
class ContinuousA2C(tf.keras.Model):
    def __init__(self, action_size):
        super(ContinuousA2C, self).__init__()
        self.actor_fc1 = Dense(24, activation='tanh')
        self.actor_mu = Dense(action_size,
                              kernel_initializer=RandomUniform(-1e-3, 1e-3))
        self.actor_sigma = Dense(action_size, activation='sigmoid',
                                 kernel_initializer=RandomUniform(-1e-3, 1e-3))

        self.critic_fc1 = Dense(24, activation='tanh')
        self.critic_fc2 = Dense(24, activation='tanh')
        self.critic_out = Dense(1,
                                kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        actor_x = self.actor_fc1(x)
        mu = self.actor_mu(actor_x)
        sigma = self.actor_sigma(actor_x)
        sigma = sigma + 1e-5

        critic_x = self.critic_fc1(x)
        critic_x = self.critic_fc2(critic_x)
        value = self.critic_out(critic_x)
        return mu, sigma, value


# 카트폴 예제에서의 연속적 액터-크리틱(A2C) 에이전트
class ContinuousA2CAgent:
    def __init__(self, action_size, max_action):
        self.render = True

        # 행동의 크기 정의
        self.action_size = action_size
        self.max_action = max_action

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        # 정책신경망과 가치신경망 생성
        self.model = ContinuousA2C(self.action_size)
        # 최적화 알고리즘 설정, 미분값이 너무 커지는 현상을 막기 위해 clipnorm 설정
        self.optimizer = Adam(lr=self.learning_rate, clipnorm=1.0)

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        mu, sigma, _ = self.model(state)
        dist = tfd.Normal(loc=mu[0], scale=sigma[0])
        action = dist.sample([1])[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            mu, sigma, value = self.model(state)
            _, _, next_value = self.model(next_state)
            target = reward + (1 - done) * self.discount_factor * next_value[0]

            # 정책 신경망 오류 함수 구하기
            advantage = tf.stop_gradient(target - value[0])
            dist = tfd.Normal(loc=mu, scale=sigma)
            action_prob = dist.prob([action])[0]
            cross_entropy = - tf.math.log(action_prob + 1e-5)
            actor_loss = tf.reduce_mean(cross_entropy * advantage)

            # 가치 신경망 오류 함수 구하기
            critic_loss = 0.5 * tf.square(tf.stop_gradient(target) - value[0])
            critic_loss = tf.reduce_mean(critic_loss)

            # 하나의 오류 함수로 만들기
            loss = 0.1 * actor_loss + critic_loss

        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))
        return loss, sigma


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500

    env = Env()
    # 환경으로부터 상태와 행동의 크기를 받아옴
    preyState_size = 8
    preyAction_size = 2
    max_prey_action = 5

    predState_size = 8
    predAction_size = 2
    max_pred_action = 5

    # 액터-크리틱(A2C) 에이전트 생성
    agent = ContinuousA2CAgent(predAction_size, max_pred_action)
    scores, episodes = [], []
    score_avg = 0
    numStep = 100

    num_episode = 1000
    for e in range(num_episode):
        done = False
        score = 0
        loss_list, sigma_list = [], []
        preyState, predState  = env.reset()

        # strate resize
        preyState = np.reshape(preyState, [1, preyState_size])
        predState = np.reshape(predState, [1, predState_size])
        
         
        if score_avg > 100:
            numStep = 300
        
        env.spawndist = 100 + e/5
         
        while not done:

            # get action
            preyAction = (1*(random.random()-0.5),1*(random.random()-0.5))
            predAction = agent.get_action(predState)

            # step
            agent.render = True #(e%10==0)
            next_preyState, preyReward, next_predState, predReward, done = env.step(preyAction,predAction,agent.render,numStep)
            
            # state resizing
            next_preyState = np.reshape(next_preyState, [1, preyState_size])
            next_predState = np.reshape(next_predState, [1, predState_size])

            # 타임스텝마다 보상 0.1, 에피소드가 중간에 끝나면 -1 보상
            score += predReward
            reward = predReward

            # 매 타임스텝마다 학습
            loss, sigma = agent.train_model(predState, predAction, predReward, next_predState, done)
            loss_list.append(loss)
            sigma_list.append(sigma)
            
            preyState = next_preyState
            predState = next_predState
            

            if done:
                # 에피소드마다 학습 결과 출력
                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print("episode: {:3d} | score avg: {:3.2f} | loss: {:.3f} | sigma: {:.3f}".format(
                      e, score_avg, np.mean(loss_list), np.mean(sigma)))

                scores.append(score_avg)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("average score")
                pylab.savefig("./save_graph/graph.png")

            if e%100 == 0:
                agent.model.save_weights("./save_model/model", save_format="tf")
