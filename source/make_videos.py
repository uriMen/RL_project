import argparse
from os.path import join, abspath, exists
from os import makedirs

import numpy as np

from utils import trace_to_video, pickle_load, multi_trace_to_video, save_args


def main(args):

    traces = pickle_load(join(args.data_load_dir, 'Traces.pkl'))
    states = pickle_load(join(args.data_load_dir, 'States.pkl'))

    # trace_to_video(traces[args.trace_idx], str(args.trace_idx),
    #                states, args.videos_output_dir, args.fps)

    # if np.sum(traces[args.traces_idx[0]].is_action_random) > 2:
    traces_to_vid = [traces[i] for i in args.traces_idx]
    multi_trace_to_video(traces_to_vid, args.video_name, states,
                         args.output_dir, args.fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make video from trace')
    parser.add_argument('--data_load_dir', help='path to existing traces',
                        type=str)
    parser.add_argument('--output_dir',
                        help='path to a directory where videos will be saved',
                        type=str)
    parser.add_argument('--fps',
                        help='how many frames-per-second in the output video',
                        type=int)
    parser.add_argument('--traces_idx',
                        help='indexes of traces to make video of',
                        type=list)
    parser.add_argument('--video_name',
                        help='a string to add to the video file name',
                        type=str)
    args = parser.parse_args()

    args.data_load_dir = abspath(
        join('..', "collected_data/episodes_with_random_actions/new_default_agent/train_data_rand_0.05_2_actions"))
    args.output_dir = abspath(
        join('..', "collected_data/results/videos_v4",
             "rand_actions"))
    if not exists(args.output_dir):
        makedirs(args.output_dir)
    args.fps = 5

    traces = pickle_load(join(args.data_load_dir, 'Traces.pkl'))
    states = pickle_load(join(args.data_load_dir, 'States.pkl'))

    # for k in range(20):
    args.traces_idx = [398, 503, 896, 298, 422, 924, 428, 151, 239, 652, 755, 150, 943, 416, 146]
    for i in args.traces_idx:
        args.video_name = f'rand_actions_{i}'
        multi_trace_to_video([traces[i]], args.video_name, states,
                             args.output_dir, args.fps)
    # main(args)
    save_args(args)
    print("DONE!!")
