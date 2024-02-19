import argparse
from os import mkdir
from os.path import abspath, join, exists

import gym
# import highway_env
import torch as T
import wandb

from agent import Agent
import numpy as np
import pandas as pd
import math
from datetime import datetime

import highway_env_local
from source.utils import save_args


def main(args):
    env = gym.make(args.env)
    repeat_train = 100
    agent = Agent(gamma=args.gamma, epsilon=args.epsilon,
                  batch_size=args.batch_size, n_actions=env.action_space.n,
                  eps_end=args.eps_end, max_mem_size=10000,
                  input_dims=[math.prod(env.observation_space.shape)],
                  lr=args.lr, repeat_train=repeat_train)

    # if resuming training of trained agent. change the name of the agent
    # change epsilon to 0.01 as well
    # agent.Q_eval.load_state_dict(
    #     T.load(abspath(join(args.output_dir, "DQN_15000_1683651154.858314.pkl")))
    # )
    # agent.Q_target.load_state_dict(
    #     T.load(abspath(join(args.output_dir, "DQN_15000_1683651154.858314.pkl")))
    # )

    scores, eps_history = [], []
    # to collect data only on right lanes use these thresholds
    right_lane_reward_thresholds = {1: 0.7, 2: 0.4, 3: 0, 4: -1}
    args_dict = {}
    for arg_name, arg_value in vars(args).items():
        args_dict[arg_name] = arg_value
    wandb.init(
        # set the wandb project where this run will be logged
        project="train_agent_process",
        # track hyper parameters and run metadata
        config=args_dict
    )

    total_num_steps = 0
    lane_resets = 0
    action_hist = [0] * 5
    for ep_idx in range(args.n_episodes):
        score = 0
        done = False
        observation, info = env.reset()
        save_next_state = True  # info['rewards']['right_lane_reward'] > right_lane_reward_thresholds[args.num_of_lanes_in_data]
        episode_length = 0
        while not done:
            observation = np.array(observation).flatten()
            action = agent.act(observation)
            action_hist[action] += 1
            new_observation, reward, done, trunc, info = env.step(action)
            total_num_steps += 1
            episode_length += 1
            # env.render()
            score += reward
            new_observation = np.array(new_observation).flatten()
            if info['rewards']['right_lane_reward'] > right_lane_reward_thresholds[args.num_of_lanes_in_data]:
                done = True
                lane_resets += 1
            agent.store_transition(observation, action, reward, new_observation, done)
            #
            # if not args.is_right_lane_training:
            #     agent.store_transition(observation, action, reward,
            #                            new_observation, done)
            # else:
            #     # save_next_state = info['rewards']['right_lane_reward'] > \
            #     #                   right_lane_reward_thresholds[
            #     #                       args.num_of_lanes_in_data]
            #     # if not save_next_state:
            #     #     done = True
            #     #if save_next_state:
            #     agent.store_transition(observation, action, reward, new_observation, done)
            #         # save_next_state = False
            #     # save_next_state = info['rewards']['right_lane_reward'] > right_lane_reward_thresholds[args.num_of_lanes_in_data]
            #
            #     # if not save_next_state:
            #     #     done = True
            # # agent.learn()  # original code, e.g., for new_default_agent
            observation = new_observation[:]
        agent.learn()  # moved from 2 line above
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])  # moving average of last 100 games

        wandb.log({"train_round": ep_idx,
                   "avg_score": score,
                   "episode_length": episode_length,
                   "total_num_steps": total_num_steps,
                   "losses": np.mean(agent.losses[-repeat_train:]),
                   "Q": np.mean(agent.qs[-repeat_train:]),
                   "lane_resets": lane_resets / (ep_idx+1)})
        wandb.log({"action_" + str(a): action_hist[a]/(ep_idx+1) for a in range(5)})

        if args.verbose:
            if (ep_idx+1) % 10 == 0:
                print(datetime.now().time(), 'episode', ep_idx+1, 'score %.2f' % score,
                      'average score %.2f' % avg_score,
                      'epsilon %.2f' % agent.epsilon)

        # save agent's DQN every 100 episodes
        if (ep_idx+1) % args.save_every == 0:
            T.save(agent.Q_target.state_dict(),
                   abspath(join(args.output_dir,
                                f'DQN_{ep_idx+1}_{datetime.timestamp(datetime.now())}.pkl')))
            # save results
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
    parser.add_argument('--is_right_lane_training',
                        help='train agent on x most right lanes data',
                        type=bool, default=False)
    parser.add_argument('--num_of_lanes_in_data',
                        help='how many lanes (from the right) to include in '
                             'training data',
                        type=int, default=4)
    parser.add_argument('--run_description',
                        help='information about the current run',
                        type=str, default="")
    args = parser.parse_args()

    # to use none default env uncomment and set value of the line below
    # args.env = "highway2-v0"
    args.gamma = 0.99
    args.epsilon = 1
    args.eps_end = 0.01
    args.lr = 0.001
    args.batch_size = 32
    args.n_episodes = 20000
    args.save_every = 500
    args.is_right_lane_training = True
    args.num_of_lanes_in_data = 3
    args.output_dir = abspath("agents/right_lanes_agent__test_9")
    args.res_file_name = "results.csv"
    args.run_description = "train on data of 3 lanes only"
    if not exists(args.output_dir):
        mkdir(args.output_dir)

    save_args(args)


    main(args)
    print("DONE!!!")


# to train the new agent:
# traces[464].obs[15] == traces[464].new_obs[14]
