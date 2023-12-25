"""
train a supervised model whose features are states and output is action.
"""
import argparse
from os.path import join, abspath

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from utils import pickle_load, create_state_action_pairs


class Model:
    def __init__(self, input_dims, max_mem_size=100000, n_actions=5,
                 epsilon=0.025):
        self.model = XGBClassifier(num_class=n_actions)
        self.mem_cntr = 0
        self.mem_size = max_mem_size
        self.input_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.output_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.encoded_actions = False
        self.actions_encoder = None
        self.epsilon = epsilon

    def store_train_data(self, traces):
        for trace in traces:
            for i in range(trace.length):
                index = self.mem_cntr % self.mem_size
                self.input_memory[index] = trace.obs[i]
                self.output_memory[index] = trace.actions[i]
                self.mem_cntr += 1
        # remove unused cells
        self.input_memory = self.input_memory[:self.mem_cntr]
        self.output_memory = self.output_memory[:self.mem_cntr]
        self.shuffle_train_data()

    def shuffle_train_data(self):
        """shuffle input and output data together"""
        assert len(self.input_memory) == len(self.output_memory)
        p = np.random.permutation(self.mem_cntr)
        self.input_memory = self.input_memory[p]
        self.output_memory = self.output_memory[p]

    def train(self):
        try:
            self.model.fit(self.input_memory, self.output_memory)
            # print(self.model.feature_importances_)
        except ValueError:
            self.encoded_actions = True
            self.actions_encoder = LabelEncoder()
            self.actions_encoder.fit(self.output_memory)
            y_train = self.actions_encoder.fit_transform(self.output_memory)
            self.model.fit(self.input_memory, y_train)
            # print(self.model.feature_importances_)

    def hyper_param_tune(self, params):
        pass

    def load_model(self, saved_model):
        """
        load json file.
        :param saved_model: path to saved model json file
        """
        self.model.load_model(saved_model)

    def transform_array(self, original_array):
        """add a column of epsilons and scale other values of each row.
        use if labels were encoded due to missing data of one label.
        """
        scaling_factor = 1 - self.epsilon
        # Apply the scaling factor to each element
        t_array = original_array * scaling_factor
        # add a column of epsilon on position 1
        t_array = np.insert(
            t_array, 1, np.array([self.epsilon] * t_array.shape[0]), axis=1)

        return t_array

    def calc_score(self, eval_states, eval_actions):
        """calculate avg log-likelihood"""
        y_prob = self.model.predict_proba(eval_states)
        if self.encoded_actions:
            y_prob = self.transform_array(y_prob)
        log_likelihood = np.log(y_prob[range(len(eval_actions)), eval_actions])
        average_log_likelihood = np.mean(log_likelihood)
        return average_log_likelihood


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='supervised model')
    parser.add_argument('--eval_data_dir',
                        help='path to evaluation data directory', type=str)
    args = parser.parse_args()

    args.train_data_dir = abspath(
        join("..",
             "collected_data/results/new_default_agent/train_data"))
    args.eval_data_dir = abspath(
        join("..",
             "collected_data/results/new_default_agent/eval_data_100_traces"))

    traces = pickle_load(join(args.train_data_dir, 'Traces.pkl'))
    states = pickle_load(join(args.train_data_dir, 'States.pkl'))

    eval_traces = pickle_load(join(args.eval_data_dir, 'Traces.pkl'))
    eval_states = pickle_load(join(args.eval_data_dir, 'States.pkl'))

    eval_states, eval_actions = list(
        zip(*create_state_action_pairs(eval_traces, eval_states, 'all')))


    """sanity checks asked by assaf"""
    # 1. not sure, verify with assaf

    # 2.
    # summary_1 = [482, 990, 546, 779, 558, 406, 360]
    # summary_2 = [654, 654, 654, 654, 654, 654, 654]
    #
    # summaries = [[482, 990, 546, 779, 558, 406, 360], [654, 654, 654, 654, 654, 654, 654]]
    #
    # scores = {0: [], 1: []}
    # for j, s in enumerate(summaries):
    #     model = Model([25])
    #     model.store_train_data([traces[i] for i in s])
    #     model.train()
    #     for t in eval_traces:
    #         scores[j].append(model.calc_score(t.obs, t.actions))
    # # print(s, "score: ", score)
    # pd.DataFrame(data=scores).to_csv("sanity_check_2.csv")


    print(np.unique(eval_actions, return_counts=True))


