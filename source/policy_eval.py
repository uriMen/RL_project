import argparse
import math
import os
import sys
from os import makedirs, path
from os.path import abspath, join

import gym
import numpy as np
import pandas as pd
import torch as T
from datetime import datetime
from random import randint

from eval_ensemble import (Ensemble, calc_score_1, calc_score_2, calc_score_3,
                           calc_score_4, calc_score_5,
                           compute_cumulative_discounted_reward)
from utils import pickle_load, pickle_save, create_state_action_pairs, save_args
from data_processing import create_df, get_traces_by_abs_diff
import highway_env_local


def train_eval_ensemble(args, traces, states, save=True):
    """return a trained Ensemble of evaluation dqn"""

    ensemble = Ensemble(args.ensemble_size, args.gamma, args.batch_size,
                        args.lr, args.input_dims, args.n_actions)

    ensemble.load_train_data(traces, states)
    ensemble.train_nets(args.train_rounds, args.output_dir, save)

    return ensemble


def load_ensemble_nets(ensemble: Ensemble, nets_dir: str):
    """
    :param ensemble:
    :param nets_dir: path to saved pkl files of nets. Number of files
    should match ensemble size.
    """
    pickles = []
    for (root, dirs, files) in os.walk(nets_dir):
        for f in files:
            if '.pkl' in f:
                pickles.append(f)

    if ensemble.size != len(pickles):
        raise ValueError("Number of pkl files doesn't match the Ensemble size.")

    for i in range(ensemble.size):
        ensemble.eval_nets[i].Q_eval.load_state_dict(
            T.load(abspath(join(nets_dir, pickles[i])))
        )


def calculate_ensemble_score(args):
    """load candidate and target ensemble and calculate the candidate
    ensemble's score relative to the target ensemble
    """
    test_traces = pickle_load(join(args.test_data_load_dir, 'Traces.pkl'))
    test_states = pickle_load(join(args.test_data_load_dir, 'States.pkl'))

    states_actions = create_state_action_pairs(test_traces, test_states)

    eval_ensemble = Ensemble(args.ensemble_size, args.gamma, args.batch_size,
                             args.lr, args.input_dims, args.n_actions)

    load_ensemble_nets(eval_ensemble, args.target_ensemble_dir)

    cand_ensemble = Ensemble(args.ensemble_size, args.gamma, args.batch_size,
                             args.lr, args.input_dims, args.n_actions)
    load_ensemble_nets(cand_ensemble, args.candidate_ensemble_dir)
    return calc_score_5(cand_ensemble, eval_ensemble, states_actions)
    # cumulative_discounted_rewards = [
    #     compute_cumulative_discounted_reward(t.rewards, args.gamma)
    #     for t in test_traces[:20]
    # ]
    # return calc_score_4(cand_ensemble, eval_ensemble, states_actions,
    #                     np.array(cumulative_discounted_rewards))


def candidates_evaluation(args):
    """draw traces, train eval ensemble and calculate scores.

    save results to a csv file.
    pre-requisites:
        1. target ensemble's nets saved in args.target_ensemble_dir
        2. test data (traces and states) saved in args.test_data_load_dir
    """
    np.random.seed(4)  # (27)
    results = []

    # load target ensemble
    target_ensemble = Ensemble(args.ensemble_size, args.gamma, args.batch_size,
                               args.lr, args.input_dims, args.n_actions)
    load_ensemble_nets(target_ensemble, args.target_ensemble_dir)

    # load train data
    train_traces = pickle_load(join(args.data_load_dir, 'Traces.pkl'))
    train_states = pickle_load(join(args.data_load_dir, 'States.pkl'))

    # load evaluation data
    eval_traces = pickle_load(join(args.test_data_load_dir, 'Traces.pkl'))
    eval_states = pickle_load(join(args.test_data_load_dir, 'States.pkl'))
    eval_states_actions = create_state_action_pairs(eval_traces, eval_states)

    for i in range(args.n_candidates):
        # draw traces
        traces_index = np.random.randint(0, len(train_traces),
                                         args.n_traces_in_summary)
        selected_traces = [train_traces[j] for j in traces_index]
        # train candidate
        cand_ensemble = train_eval_ensemble(args, selected_traces,
                                            train_states, save=False)
        # for calc_score_4
        selected_traces_states_actions = create_state_action_pairs(
            selected_traces, train_states)
        cumulative_discounted_rewards = [
            compute_cumulative_discounted_reward(t.rewards, args.gamma)
            for t in selected_traces
        ]

        candidate_info = {
            'used_traces': traces_index.tolist(),
            's1': calc_score_1(cand_ensemble, target_ensemble,
                               eval_states_actions),
            's2': calc_score_2(cand_ensemble, target_ensemble,
                               eval_states_actions),
            's3': calc_score_3(cand_ensemble, target_ensemble,
                               eval_states_actions),
            's4': calc_score_4(target_ensemble, selected_traces_states_actions,
                               cumulative_discounted_rewards),
            's5': calc_score_5(cand_ensemble, target_ensemble,
                               eval_states_actions),
        }
        results.append(candidate_info)
        print(f'{datetime.now()} Finished candidate {i+1}')
        if (i + 1) % 50 == 0:
            df = pd.DataFrame(data=results)
            df.to_csv(abspath(join(args.output_dir, 'candidates_score.csv')))

    df = pd.DataFrame(data=results)
    df.to_csv(abspath(join(args.output_dir, 'candidates_score.csv')))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='policy evaluation')
    parser.add_argument('--data_load_dir', help='path to existing traces',
                        type=str)
    parser.add_argument('--test_data_load_dir', help='path to existing traces',
                        type=str)
    parser.add_argument('--output_dir', help='path to dir for saving nets',
                        type=str)
    parser.add_argument('--target_ensemble_dir',
                        help='path to dir of saved nets for target ensemble',
                        type=str)
    parser.add_argument('--candidate_ensemble_dir',
                        help='path to dir of saved nets for candidate ensemble',
                        type=str)
    parser.add_argument('--gamma', help='discount factor', type=float)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--n_actions', help='number of actions', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--input_dims', help='input dimensions', type=list)
    parser.add_argument('--ensemble_size', help='number of nets in ensemble',
                        type=float)
    parser.add_argument('--train_rounds', help='number of train rounds',
                        type=int)
    parser.add_argument('--n_candidates',
                        help='number candidates to evaluate', type=int)
    parser.add_argument('--n_traces_in_summary',
                        help='number of traces to use for creating a candidate',
                        type=int)
    parser.add_argument('--env', help='environment to train in',
                        type=str, default='highway-v0')
    args = parser.parse_args()

    args.data_load_dir = abspath(
        join('..', "collected_data/results/new_bad_agent/train_data"))
    # args.test_data_load_dir = abspath(
    #     join('..', "collected_data/results/test_set"))
    # args.target_ensemble_dir = abspath(
    #     join('..', "eval ensemble/target ensemble"))
    # args.candidate_ensemble_dir = abspath(
    #     join('..', "eval ensemble/20_random_selected"))
    log_name = f'run_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_' \
               f'{randint(100000, 900000)}'
    args.output_dir = abspath(
        join(args.data_load_dir, "eval ensemble"))
    if not path.exists(args.output_dir):
        makedirs(args.output_dir)
    args.gamma = 0.99
    args.lr = 0.001
    args.batch_size = 32
    args.ensemble_size = 5
    args.train_rounds = 50000
    args.n_candidates = 1000
    # args.n_traces_in_summary = 10

    # for none default env uncomment bellow
    args.env = "highway-v0"

    env = gym.make(args.env)

    args.input_dims = [math.prod(env.observation_space.shape)]
    args.n_actions = env.action_space.n

    # save args to file
    save_args(args)

    traces = pickle_load(join(args.data_load_dir, 'Traces.pkl'))
    states = pickle_load(join(args.data_load_dir, 'States.pkl'))

    train_eval_ensemble(args, traces, states)
    # r = candidates_evaluation(args)

    print("DONE!!")
