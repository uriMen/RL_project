import argparse
from os import mkdir
from os.path import abspath, join, exists

import gym
# import highway_env
import torch as T

from agent import Agent
import numpy as np
import pandas as pd
import math
from datetime import datetime

import highway_env_local
from source.utils import save_args


def main(args):
    env = gym.make(args.env)
    agent = Agent(gamma=args.gamma, epsilon=args.epsilon,
                  batch_size=args.batch_size, n_actions=env.action_space.n,
                  eps_end=args.eps_end,
                  input_dims=[math.prod(env.observation_space.shape)],
                  lr=args.lr)

    # if resuming training of trained agent. change the name of the agent
    # change epsilon to 0.01 as well
    # agent.Q_eval.load_state_dict(
    #     T.load(abspath(join(args.output_dir, "DQN_15000_1683651154.858314.pkl")))
    # )
    # agent.Q_target.load_state_dict(
    #     T.load(abspath(join(args.output_dir, "DQN_15000_1683651154.858314.pkl")))
    # )

    scores, eps_history = [], []

    for i in range(args.n_episodes):
        score = 0
        done = False
        observation, info = env.reset()
        while not done:
            observation = np.array(observation).flatten()
            action = agent.act(observation)
            observation_, reward, done, trunc, info = env.step(action)
            env.render()
            score += reward
            observation_ = np.array(observation_).flatten()
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])  # moving average of previous 100 games

        if args.verbose:
            if (i+1) % 10 == 0:
                print(datetime.now().time(), 'episode', i+1, 'score %.2f' % score,
                      'average score %.2f' % avg_score,
                      'epsilon %.2f' % agent.epsilon)

        # save agent's DQN every 100 episodes
        if (i+1) % args.save_every == 0:
            T.save(agent.Q_target.state_dict(),
                   abspath(join(args.output_dir,
                                f'DQN_{i+1}_{datetime.timestamp(datetime.now())}.pkl')))
            results = pd.DataFrame(
                data={'scores': scores, "epsilon": eps_history})
            results.to_csv(abspath(join(args.output_dir, args.res_file_name)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train agent')
    parser.add_argument('--env', help='environment to train in',
                        type=str, default='highway-v0')
    parser.add_argument('-v', '--verbose',
                        help='print information to the console',
                        action='store_true', default=True)
    parser.add_argument('--output_dir',
                        help='path to dir where trained agents are saved in',
                        type=str)
    parser.add_argument('--res_file_name',
                        help='results file name',
                        type=str)
    parser.add_argument('--n_episodes', help='number of episodes to train',
                        type=int)
    parser.add_argument('--save_every',
                        help='number of episodes between saving an agent',
                        type=int)
    parser.add_argument('--gamma', help='discount factor', type=float)
    parser.add_argument('--epsilon', help='hyper parameter', type=float)
    parser.add_argument('--eps_end', help='hyper parameter', type=float)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--batch_size', help='training batch size', type=float)
    args = parser.parse_args()

    # to use none default env uncomment and set value of the line below
    args.env = "highway2-v0"
    args.gamma = 0.99
    args.epsilon = 1
    args.eps_end = 0.01
    args.lr = 0.001
    args.batch_size = 32
    args.n_episodes = 20000
    args.save_every = 500
    args.output_dir = abspath(join("..", "agents/new_different_agent"))
    args.res_file_name = "results.csv"
    if not exists(args.output_dir):
        mkdir(args.output_dir)

    save_args(args)

    main(args)
    print("DONE!!!")
