import argparse
import math
from os.path import join, abspath

import numpy as np
import pandas as pd
import torch as T

from utils import pickle_load, compute_cumulative_discounted_reward


def create_df(traces, states, args):
    """create a DataFrame from collected data"""
    data = []
    for trace in traces:
        # state_info = {
        #     0: 'initial',
        #     trace.length: 'terminal',
        #     'else': 'internal'
        # }
        for i in range(trace.length):
            row = {
                "state_id": trace.states[i],
                "q_value": T.max(
                    states[trace.states[i]].observed_actions).item(),
                "discounted_reward": compute_cumulative_discounted_reward(
                    trace.rewards[i:], args.gamma),
                "state_info": "initial" if i == 0 else "internal"
            }
            data.append(row)
        terminal_state_id = (trace.states[0][0], trace.length)
        terminal_state_row = {
            "state_id": terminal_state_id,
            "q_value": T.max(
                states[terminal_state_id].observed_actions).item(),
            "discounted_reward": T.max(
                states[terminal_state_id].observed_actions).item(),
            "state_info": "terminal"
        }
        data.append(terminal_state_row)

    df = pd.DataFrame(data)
    df['q_difference'] = df['q_value'] - df['discounted_reward']
    # save to file
    # df.to_csv(abspath(join(args.load_dir, 'data_1.csv')))
    return df


def get_traces_by_abs_diff(states_df: pd.DataFrame, best=True, n=20) -> list:
    """given the df created in create_df func, return the ids of the
    traces with the smallest or largest absolute difference between init
    q value and cumulative discounted reward. If 'best' is True, return
    the traces with the smallest abs diff.
    """
    states_df['abs_diff'] = np.absolute(states_df['q_difference'].values)
    df = states_df[states_df['state_info'] == 'initial'].sort_values(
        'abs_diff', ascending=best).head(n)

    return [x[0] for x in df['state_id'].values]


def data_prep_for_supervised_learning(traces):
    """
    Prepare data for Q-func estimation.
    :param traces: list of Trace objects.
    :return: features array (obs), labels array (action) for estimating
     a network.
    """
    features = []
    labels = []
    for trace in traces:
        for i in range(trace.length):
            features.append(trace.obs[i])
            labels.append(trace.actions[i])

    return features, labels


def print_init_states(traces, states):
    """print initial state of each trace and its final reward"""
    for t in traces:
        init_state = states[t.states[0]]
        init_state.plot_image()
        print("final reward: ", t.reward_sum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='see_data')
    parser.add_argument('--load_dir', help='path to existing traces', type=str)
    parser.add_argument('--gamma', help='discount factor', type=float)
    args = parser.parse_args()

    args.load_dir = abspath(
        join('..', "collected_data/results/first_default_agent_data_1000_traces"))
    args.gamma = 0.5

    traces = pickle_load(join(args.load_dir, 'Traces.pkl'))
    states = pickle_load(join(args.load_dir, 'States.pkl'))

    df = create_df(traces, states, args)
    print(get_traces_by_abs_diff(df))
    # X, y = data_prep_for_supervised_learning(traces[:2])

    # print(traces[1].states, traces[1].actions)
    # for t in traces[:10]:
        # print(t.length, t.dones[-1])
    # print(len(states))
    # t = T.tensor(states[(2, 3)].observed_actions)
    # print(t)
    print("DONE!!")

