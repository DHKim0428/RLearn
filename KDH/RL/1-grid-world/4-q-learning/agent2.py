import numpy as np
import random
from collections import defaultdict
from environment import Env


class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.step_size = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.3
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        state, next_state = str(state), str(next_state)
        current_q = self.q_table[state][action]
        max_q = max(self.q_table[next_state])
        td = reward + self.discount_factor * max_q - current_q
        new_q = current_q + self.step_size * td
        self.q_table[state][action] = new_q
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state = str(state)
            q_list = self.q_table[state]
            action = arg_max(q_list)
        return action


def arg_max(q_list):
    max_idx_list = np.argwhere(q_list == np.amax(q_list))
    max_idx_list = max_idx_list.flatten().tolist()
    return random.choice(max_idx_list)

if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()

        while True:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)

            state = next_state

            env.print_value_all(agent.q_table)

            if done:
                break