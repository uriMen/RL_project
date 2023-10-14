import argparse
from os.path import join, abspath, exists
from os import makedirs
import math
from datetime import datetime
from random import randint

import gym
import highway_env
import torch as T
import numpy as np
import pandas as pd

from agent import Agent
import get_traces
from utils import pickle_save, pickle_load, save_args, make_clean_dirs
from eval_ensemble import EvalNet
import highway_env_local


def make_environment(env_name):
    return gym.make(env_name)


def load_agent(args, is_eval_net=False):

    if is_eval_net:
        agent = EvalNet(gamma=args.gamma, batch_size=args.batch_size,
                        n_actions=env.action_space.n,
                        input_dims=[math.prod(env.observation_space.shape)],
                        lr=args.lr)
        agent.Q_eval.load_state_dict(
            T.load(args.eval_net_path))
    else:
        agent = Agent(gamma=args.gamma, epsilon=args.epsilon,
                      batch_size=args.batch_size,
                      n_actions=env.action_space.n, eps_end=args.eps_end,
                      input_dims=[math.prod(env.observation_space.shape)],
                      lr=args.lr)

        agent.Q_eval.load_state_dict(
            T.load(args.agent_path))

    return agent


def compare_agents(org_agent, est_agent, env, args):
    """compare q values of original agent vs estimated agent"""
    q_diff_selected_action = []
    q_diff_other_actions = []
    for _ in range(args.n_traces):
        curr_obs, info = env.reset()
        done = False
        while not done:
            curr_obs = np.array(curr_obs).flatten()
            """take an action and update environment"""
            a = org_agent.act(curr_obs)
            obs, r, done, trunc, infos = env.step(a)
            # if done then r = r + max(state_q_values)
            obs = np.array(obs).flatten()

            state_q_values_org = org_agent.get_state_action_values(curr_obs)
            state_q_values_est = est_agent.get_state_action_values(curr_obs)

            abs_diff = T.abs(state_q_values_org - state_q_values_est)

            q_diff_selected_action.append(abs_diff[a].item())
            # tensor[tensor != tensor[index_to_exclude]]
            q_diff_other_actions += abs_diff[abs_diff != abs_diff[a]].tolist()

            curr_obs = obs

    np.savetxt(abspath(join(args.output_dir, "selected_action.csv")),
               q_diff_selected_action, delimiter=",")
    np.savetxt(abspath(join(args.output_dir, "other_actions.csv")),
               q_diff_other_actions, delimiter=",")
    # return q_diff_selected_action, q_diff_other_actions


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='collect_data')
    parser.add_argument('-n', '--n_traces', help='number of traces to obtain',
                        type=int, default=10)
    parser.add_argument('-v', '--verbose',
                        help='print information to the console',
                        action='store_true', default=True)
    parser.add_argument('-ap', '--agent_path',
                        help='path to load saved agent', type=str)
    parser.add_argument('--eval_net_path',
                        help='path to load saved evaluation net', type=str)
    parser.add_argument('--env', help='environment to train in',
                        type=str, default='highway-v0')
    parser.add_argument('--gamma', help='discount factor', type=float)
    parser.add_argument('--epsilon', help='hyper parameter', type=float)
    parser.add_argument('--eps_end', help='hyper parameter', type=float)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--batch_size', help='training batch size', type=float)
    parser.add_argument('--rand_rate',
                        help='for trained agent only. use for debug.'
                             'agent will take random actions at this rate',
                        type=float, default=0)
    parser.add_argument('--n_rand_actions',
                        help='use with rand_rate. if a random action is taken'
                             'agent will act it for this number of steps',
                        type=int, default=0)
    args = parser.parse_args()

    log_name = f'run_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_' \
               f'{randint(100000, 900000)}'

    # default agent
    # args.agent_path = abspath(
    #     join('..', 'agents/new_default_agent/DQN_20000_1683742777.213296.pkl'))
    # different agent
    args.agent_path = abspath(
        join('..', 'agents/new_different_agent/DQN_20000_1683933469.517535.pkl'))

    args.output_dir = abspath(join('..', 'collected_data/results/new_different_agent/eval_data_100_traces'))
    # eval_net
    # args.eval_net_path = abspath(
    #     join('..', 'eval ensemble/net.pkl'))

    # to use none default env uncomment and set value of the line below
    args.env = "highway2-v0"

    args.gamma = 0.99
    args.epsilon = 0
    args.eps_end = 0.01
    args.lr = 0.001
    args.batch_size = 32
    args.rand_rate = 0 # 0.05  # create episodes with random actions
    args.n_rand_actions = 1

    if not exists(args.output_dir):
        makedirs(args.output_dir)

    args.n_traces = 100

    env = make_environment(args.env)
    agent = load_agent(args, is_eval_net=False)
    # eval_agent = load_agent(args, is_eval_net=True)

    # save args to file
    save_args(args)

    # to collect data run 3 lines below
    traces, states = get_traces.get_traces(env, agent, args)
    pickle_save(traces, join(abspath(args.output_dir), 'Traces.pkl'))
    pickle_save(states, join(abspath(args.output_dir), 'States.pkl'))

    # to compare agent and eval_net run code below
    # compare_agents(agent, eval_agent, env, args)

    print("DONE!!")
    env.close()
