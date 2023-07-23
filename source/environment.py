import math
import gym
import highway_env
from matplotlib import pyplot as plt
import torch as T
import numpy as np

from agent import Agent, DQN


env = gym.make('highway-v0')
# print(env.observation_space.shape)
# print(env.action_space)

# dqn = DQN(0.01, input_dims=[math.prod(env.observation_space.shape)], fc1_dims=256,
#           fc2_dims=256, n_actions=env.action_space.n)
#
# dqn.load_state_dict(T.load('agents/DQN_200_1667555578.890477.pkl'))
#                            # map_location=dqn.device))

my_agent = Agent(gamma=0.99, epsilon=0.01, batch_size=16,
                 n_actions=env.action_space.n, eps_end=0.01,
                 input_dims=[math.prod(env.observation_space.shape)], lr=0.001)
my_agent.Q_eval.load_state_dict(T.load('agents/DQN_150_1667554420.433796.pkl'))
                                       # map_location=my_agent.Q_eval.device))

# play with agent
for i in range(10):
    observation = env.reset()
    score = 0
    done = False
    while not done:
        observation = np.array(observation).flatten()
        action = my_agent.act(observation)
        observation_, reward, done, info = env.step(action)
        env.render()
        score += reward
        observation_ = np.array(observation_).flatten()
        observation = observation_

    print('episode', i+1, 'score %.2f' % score)


# for _ in range(10):
#     action = env.action_type.actions_indexes["IDLE"]
#     obs, reward, done, info = env.step(action)
#     env.render()
#
# plt.imshow(env.render(mode="rgb_array"))
# plt.show()