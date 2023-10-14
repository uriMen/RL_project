import os
import glob
import pickle
from os.path import join, abspath

import av
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np


class Trace(object):
    def __init__(self):
        self.obs = []
        self.new_obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.reward_sum = 0
        self.game_score = None
        self.length = 0
        self.states = []
        self.is_action_random = []

    def update(self, obs, new_obs, r, done, infos, a, state_id,
               is_action_random=False):
        self.obs.append(obs)
        self.new_obs.append(new_obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.actions.append(a)
        self.reward_sum += r
        self.states.append(state_id)
        self.length += 1
        self.is_action_random.append(is_action_random)


class State(object):
    def __init__(self, name, obs, action_vector, feature_vector, img):
        self.observation = obs
        self.image = img
        self.observed_actions = action_vector  # state's q values
        self.name = name
        self.features = feature_vector

    def plot_image(self):
        plt.imshow(self.image)
        plt.show()

    def save_image(self, path, name):
        imageio.imwrite(path + '/' + name + '.png', self.image)


def pickle_load(filename):
    return pickle.load(open(filename, "rb"))


def pickle_save(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def make_clean_dirs(path, no_clean=False, file_type=''):
    try:
        os.makedirs(path)
    except:
        if not no_clean: clean_dir(path, file_type)


def clean_dir(path, file_type=''):
    files = glob.glob(path + "/*" + file_type)
    for f in files:
        os.remove(f)


def save_args(args):
    with open(abspath(join(args.output_dir, 'commandline_args.txt')), 'w') as f:
        for arg_name, arg_value in vars(args).items():
            f.write(f"{arg_name}: {arg_value}\n")


def create_video(frames_dir, output_dir, n_hls, size, fps):
    make_clean_dirs(output_dir)
    for hl in range(n_hls):
        hl_str = str(hl) if hl > 9 else "0" + str(hl)
        img_array = []
        file_list = sorted(
            [x for x in glob.glob(frames_dir + "/*.png") if x.split('/')[-1].startswith(hl_str)])
        for f in file_list:
            img = cv2.imread(f)
            img_array.append(img)
        out = cv2.VideoWriter(join(output_dir, f'HL_{hl}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


def trace_to_video(trace: Trace, trace_name: str, states: dict,
                   output_dir: str, fps: int):
    """create a video from a trace and its states"""
    frames_dir = join(output_dir, f'trace_{trace_name}_frames')
    make_clean_dirs(frames_dir)
    for s_id in trace.states:
        s_name = str(s_id[1]) if s_id[1] > 9 else f'0{str(s_id[1])}'
        states[s_id].save_image(frames_dir, s_name)
    s_name = str(s_id[1] + 1) if s_id[1] + 1 > 9 else f'0{str(s_id[1] + 1)}'
    states[(s_id[0], s_id[1] + 1)].save_image(frames_dir, s_name)
    file_list = sorted([f for f in glob.glob(frames_dir + "/*.png")])
    img_array = [cv2.imread(f) for f in file_list]
    height, width, _ = img_array[0].shape
    out = cv2.VideoWriter(join(output_dir, f'trace_{trace_name}.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def multi_trace_to_video(traces: list, name: str, states: dict,
                         output_dir: str, fps: int):
    """create a video from a list of traces"""

    frames_dir = join(output_dir, f'{name}_frames')
    img_array = []
    fade_duration = 0.7  # seconds of fade between traces. default 0.7
    fade_frames = int(fade_duration * fps)
    for trace in traces:
        make_clean_dirs(frames_dir)
        for s_id in trace.states:
            s_name = f'{s_id[0]}_{s_id[1]}' if s_id[1] > 9 \
                else f'{s_id[0]}_0{s_id[1]}'
            states[s_id].save_image(frames_dir, s_name)
        s_name = f'{s_id[0]}_{s_id[1] + 1}' if s_id[1] + 1 > 9 \
            else f'{s_id[0]}_0{s_id[1] + 1}'
        states[(s_id[0], s_id[1] + 1)].save_image(frames_dir, s_name)

        file_list = sorted([f for f in glob.glob(frames_dir + "/*.png")])
        img_array += [cv2.imread(f) for f in file_list]
        # if multiple traces add fade out between traces
        if len(traces) > 1:
            for i in range(fade_frames):
                frame = cv2.imread(file_list[-1])
                alpha = 1.0 - (i + 1) / fade_frames
                fade = np.zeros(frame.shape, dtype=np.uint8)  # np.ones(frame.shape, dtype=np.uint8) * 255
                frame = cv2.addWeighted(frame, alpha, fade, 1 - alpha, 0)
                img_array.append(frame)

    height, width, _ = img_array[0].shape
    output = av.open(join(output_dir, f'{name}.mp4'), 'w')
    stream = output.add_stream('h264', str(fps))
    stream.bit_rate = 8000000
    stream.height = height
    stream.width = width
    for i, img in enumerate(img_array):
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        packet = stream.encode(frame)
        output.mux(packet)
    # flush
    packet = stream.encode(None)
    output.mux(packet)
    output.close()


def create_state_action_pairs(traces, states, which_states='init') -> list:
    """

    :param traces: list of Trace objects.
    :param states: dictionary with State objects as values.
    :param init_only: bool. If True, using only initial state
    :param which_states: str. {'init', 'all', 'rand'}, default: 'init'.
    which state to use from each trace: 'init' for initial only, 'all' for all
    states, 'rand' for 1 random state.
    from each trace, else, using all states in each trace.
    :return: list of (observation, action) tuples.
    """
    states_actions = []
    for trace in traces:
        if which_states == 'init':
            s = states[trace.states[0]].observation
            a = trace.actions[0]
            states_actions.append((s, a))
        elif which_states == 'all':
            for i in range(trace.length):
                s = states[trace.states[i]].observation
                a = trace.actions[i]
                states_actions.append((s, a))
        elif which_states == 'rand':
            j = np.random.randint(0, trace.length)
            s = states[trace.states[j]].observation
            a = trace.actions[j]
            states_actions.append((s, a))
        else:
            raise Exception("which_state param should be one of {'init', 'all', 'rand'}")


    return states_actions


def compute_cumulative_discounted_reward(rewards: list, gamma: float) -> float:
    """
    Given a list of rewards (of each of the following states,
    compute the accumulated discounted reward.
    :param rewards: list of rewards, each is a float.
    :param gamma: discount factor.
    :return: accumulated discounted reward
    """
    acc_reward = 0
    for i, r in enumerate(rewards):
        acc_reward += (gamma ** i) * r
    return acc_reward
