import argparse
import math
import time
from os import makedirs
from os.path import abspath, join, exists

import numpy as np
import gym
import torch as T
import highway_env
from pynput import keyboard

from agent import Agent
import highway_env_local

PAUSED = False
ACTION = 1


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


def on_release(key):
    global PAUSED
    global ACTION
    print('{0} released'.format(
        key))
    if key == keyboard.Key.space:
        PAUSED = True
    elif key == "r":
        PAUSED = False
    elif key == keyboard.Key.up:
        ACTION = 0
    elif key == keyboard.Key.down:
        ACTION = 2
    elif key == keyboard.Key.left:
        ACTION = 4
    elif key == keyboard.Key.right:
        ACTION = 3
    elif key == keyboard.Key.esc:
        # Stop listener
        return False


def make_environment(env_name):
    return gym.make(env_name)


def load_agent(args, opt=False):
    """:param opt: bool. To load optimal agent"""
    agent = Agent(gamma=args.gamma, epsilon=args.epsilon,
                  batch_size=args.batch_size,
                  n_actions=env.action_space.n, eps_end=args.eps_end,
                  input_dims=[math.prod(env.observation_space.shape)],
                  lr=args.lr)

    if not opt:
        agent.Q_eval.load_state_dict(
            T.load(args.agent_path))
    else:
        agent.Q_eval.load_state_dict(
            T.load(args.opt_agent_path))

    return agent


def simulate_termination(env, subj_agent, opt_agent, args):
    """Simulate a single trace"""
    global PAUSED
    global ACTION
    curr_obs, info = env.reset()
    done = False
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    while not done:
        env.render()
        curr_obs = np.array(curr_obs).flatten()
        """take an action and update environment"""
        if not PAUSED:
            a = subj_agent.act(curr_obs)
        else:
            a = opt_agent.act(curr_obs)  # np.random.choice([0, 2], 1)[0]
        # a = ACTION
        obs, r, done, trunc, infos = env.step(a)
        obs = np.array(obs).flatten()

        """Update observation to the following one"""
        curr_obs = obs
        # ACTION = 1

    listener.stop()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='simulate termination')
    parser.add_argument('-n', '--n_traces', help='number of traces to obtain',
                        type=int, default=10)
    parser.add_argument('-v', '--verbose',
                        help='print information to the console',
                        action='store_true', default=True)
    parser.add_argument('-ap', '--agent_path',
                        help='path to load saved agent', type=str)
    parser.add_argument('--opt_agent_path',
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
    args = parser.parse_args()

    args.agent_path = abspath(
        join('..', 'agents/new_default_agent/DQN_20000_1683742777.213296.pkl'))
    args.opt_agent_path = abspath(
        join('..', 'agents/new_different_agent/DQN_20000_1683933469.517535.pkl'))
    # args.output_dir = abspath(join('..', 'collected_data/results/new_different_agent',

    args.gamma = 0.99
    args.epsilon = 0  # means no random actions at all
    args.eps_end = 0.01
    args.lr = 0.001
    args.batch_size = 32

    # if not exists(args.output_dir):
    #     makedirs(args.output_dir)

    args.n_traces = 50

    # to use none default env uncomment and set value of the line below
    args.env = "highway-v0-terminator"

    env = make_environment(args.env)
    agent = load_agent(args)
    opt_agent = load_agent(args)

    simulate_termination(env, agent, opt_agent, args)
    env.close()
    print("DONE!!")
