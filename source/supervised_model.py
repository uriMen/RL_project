"""
train a supervised model whose features are states and output is action.
"""
import argparse
import datetime
from os.path import join, abspath

import gym
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

    def act(self, state):
        """given a state, return an action"""
        action = self.model.predict((state,))
        if self.encoded_actions:
            action = self.actions_encoder.inverse_transform(action)
        return action[0]

    def get_episode_states_actions(self, env):
        """execute an episode with the model as agent
        and return its states and taken actions"""
        states = []
        actions = []
        done = False
        curr_obs, info = env.reset()
        while not done:
            # env.render(mode='rgb_array')
            curr_obs = np.array(curr_obs).flatten()
            action = self.act(curr_obs)
            states.append(curr_obs)
            actions.append(action)
            new_obs, r, done, trunc, infos = env.step(action)
            curr_obs = new_obs

        return states, actions

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

    # eval_traces = pickle_load(join(args.eval_data_dir, 'Traces.pkl'))
    # eval_states = pickle_load(join(args.eval_data_dir, 'States.pkl'))

    # eval_states, eval_actions = list(
    #     zip(*create_state_action_pairs(eval_traces, eval_states, 'all')))

    summaries = {"5_best": ([719, 628, 958, 131, 212],[413, 483, 553]),
                 "5_med": ([955, 798, 513, 860, 280], [227, 376, 835]),
                 "5_bad": ([902, 776, 28, 828, 191], [954, 992, 120]),
                 "3_best": ([759, 214, 426], [413, 483, 553]),
                 "3_med": ([743, 772, 617], [227, 376, 835]),
                 "3_bad": ([569, 704, 345], [954, 992, 120])
                 }
    summary_episodes_scores = dict()
    summary_questions_scores = []
    for summary_type, traces_idx in summaries.items():

        scores = []
        model = Model([25])
        model.store_train_data([traces[i] for i in traces_idx[0]])
        model.train()
        env = gym.make('highway-v0')
        for i in range(1000):
            states, actions = model.get_episode_states_actions(env)
            scores.append(model.calc_score(states, actions))
            print(summary_type, i)
        summary_episodes_scores[summary_type] = scores

        for idx in traces_idx[1]:
            eval_states = traces[idx].obs
            eval_actions = traces[idx].actions
            score = model.calc_score(eval_states, eval_actions)
            summary_questions_scores.append({"summary_type": summary_type,
                                             "episode_id": idx,
                                             "score": score})
        print(f'{summary_type} done {datetime.datetime.now()}')

    df = pd.DataFrame.from_dict(summary_episodes_scores)#, orient='index') #.to_csv('coputational_validation.csv')
    df_ = pd.DataFrame.from_dict(summary_questions_scores)

    df.to_csv('summary_based_episodes.csv')
    df_.to_csv('scores_of_questions.csv')
    # print(df, df_)


