"""
An evolutionary algorithm which is used to find the best summary,
based on a given score.
        notation:
            1. a gene would be a single trace
            2. a chromosome would be a set of traces which is
            a summary candidate.
            3. fitness would be the score of a chromosome which is based
            on a given score function


What is needed:
1. collect data for the agent. let's say 1000 traces.
2. train a target ensemble on all the data (Q_K)
3. draw summaries, i.e., sets of 10(?) traces each. try to make sure each trace
is contained in at least 2 summaries. this will be the 1st generation.
4. for each generation:
    a. compute fitness of each summary:
        1. train an ensemble (Q_C)
        2. calculate its score using a score function
    b. evolve, i.e., build the next generation, with the following:
        1. the most fitted summaries from the previous generation
        2. children of the most fitted summaries from the previous generation
5. repeat from line 4 until "convergence"
6. repeat from line 4 for 2 different score functions: score_2 and score_5

"""
import argparse
import math
import random
import threading
from os import makedirs
from os.path import abspath, join, exists
from datetime import datetime

import gym
import numpy as np
import pandas as pd
import torch
import wandb

from eval_ensemble import Ensemble, calc_score_2, calc_score_5, calc_q_values, \
    calc_score_5_mean, calc_score_2_mean
from utils import create_state_action_pairs, pickle_load, save_args
from policy_eval import load_ensemble_nets

# global variables
TRAIN_DATA = None
EVAL_DATA = None
EVAL_STATE_ACTION_PAIRS = None
REF_ENSEMBLE = None


class Individual:
    """An individual element in a generation"""

    def __init__(self, traces_idx: list, args):
        self.chromosome = traces_idx
        self.ensemble = Ensemble(args.ensemble_size, args.gamma,
                                 args.batch_size, args.lr, args.input_dims,
                                 args.n_actions)
        self.fitness_score = 0

    def calc_fitness(self, args):
        if args.fitness_func == "reward_sum":
            rewards = [TRAIN_DATA["traces"][gene].reward_sum for gene in self.chromosome]
            self.fitness_score = -np.sum(rewards)  # negative sign to make lower score better
        else:
            train_traces = [TRAIN_DATA["traces"][gene] for gene in self.chromosome]
            self.ensemble.load_train_data(train_traces, TRAIN_DATA["states"])
            self.ensemble.train_nets(args.train_rounds, None, False)
            self.fitness_score = args.fitness_func(
                self.ensemble, REF_ENSEMBLE, EVAL_STATE_ACTION_PAIRS)
            # self.fitness_score = np.random.randint(0, 100)

    def validation_q_value(self, args, init_only=True):
        train_traces = [TRAIN_DATA["traces"][gene] for gene in self.chromosome]
        self.ensemble.load_train_data(train_traces, TRAIN_DATA["states"])
        self.ensemble.train_nets(args.train_rounds, None, False)
        return calc_q_values(
            self.ensemble, REF_ENSEMBLE, EVAL_STATE_ACTION_PAIRS, init_only)

    def mutate_chromosome(self, args):
        """mutates an individual by randomly changing some of its genes."""
        for i in range(len(self.chromosome)):
            if random.random() < args.mutation_rate:
                self.chromosome[i] = np.random.choice(np.delete(
                    np.arange(len(TRAIN_DATA["traces"])),
                    self.chromosome), 1)


def generate_initial_population(args) -> list:
    """This function generates the initial population."""
    min_occurrence = 3
    numbers = np.arange(len(TRAIN_DATA["traces"]))
    candidates = np.zeros((args.population_size, args.chromosome_size),
                          dtype=int) - 1
    for i in range(candidates.shape[0]):
        if numbers.shape[0] < args.chromosome_size:
            candidates[i] = np.concatenate(
                (numbers,
                 np.random.choice(np.delete(
                     np.arange(len(TRAIN_DATA["traces"])), numbers),
                     args.chromosome_size - numbers.shape[0]))
            )
        else:
            candidates[i] = np.random.choice(numbers, args.chromosome_size,
                                             replace=False)
        unique, counts = np.unique(candidates, return_counts=True)
        hist = np.asarray((unique, counts)).T
        to_remove = hist[hist[:, 1] >= min_occurrence][:, 0]
        numbers = numbers[np.logical_not(np.isin(numbers, to_remove))]

    # verify each trace is used in at least min_occurrence candidates
    assert hist.min(axis=0)[1] >= min_occurrence
    print(hist.min(axis=0)[1])
    population = [Individual(candidate, args) for candidate in candidates]
    return population


def crossover(individual1, individual2):
    """performs a single-point crossover between two individuals."""
    crossover_point = random.randint(1, len(individual1.chromosome) - 1)
    chrom_1 = np.concatenate((individual1.chromosome[:crossover_point],
                              individual2.chromosome[crossover_point:]))
    chrom_2 = np.concatenate((individual2.chromosome[:crossover_point],
                              individual1.chromosome[crossover_point:]))
    child_1 = Individual(chrom_1, args)
    child_2 = Individual(chrom_2, args)
    return child_1, child_2


def select_parents(population):
    """This function selects two parents from the population.

    selection is based on fitness score - the lower fitness_score is
    the higher probability to be selected.
    """
    sorted_population = sorted(population,
                               key=lambda individual: individual.fitness_score)
    ranks = list(range(1, len(sorted_population) + 1))
    probabilities = sorted([rank / sum(ranks) for rank in ranks], reverse=True)
    parents = np.random.choice(sorted_population, 2, p=probabilities,
                               replace=False)
    return parents


def evolve(population):
    """This function creates the next generation's population."""
    auto_advance_rate = 0.1  # individuals to advance automatically to next gen
    new_population = []
    sorted_pop = sorted(population, key=lambda x: x.fitness_score)
    new_population += sorted_pop[:int(len(population) * auto_advance_rate)]
    for _ in range((args.population_size - len(new_population)) // 2):
        parent1, parent2 = select_parents(population)
        child1, child2 = crossover(parent1, parent2)
        child1.mutate_chromosome(args)
        child2.mutate_chromosome(args)
        new_population.append(child1)
        new_population.append(child2)
    return new_population


def save_population(population, idx):
    """save population of the idxth generation to a csv file.

    saves indexes of traces contained in each individual and its score.
    """
    print(f"saving population gen {idx}...")
    data = {
        "candidate": [individual.chromosome for individual in population],
        "score": [individual.fitness_score for individual in population]
    }
    df = pd.DataFrame(data=data)
    print(f'gen {idx}: avg: {df["score"].mean()}, best: {df["score"].min()}')
    df.to_csv(abspath(join(args.output_dir, f"gen_{idx}.csv")))


def load_population(args):
    df = pd.read_csv(args.pop_file, index_col=0)
    population = []
    for index, row in df.iterrows():
        if row['candidate'][1:-1].find(",") == -1:
            traces_idx = [int(i) for i in row['candidate'][1:-1].split()]
        else:
            traces_idx = [int(i) for i in row['candidate'][1:-1].split(",")]
        ind = Individual(traces_idx, args)
        ind.fitness_score = row['score']
        population.append(ind)
    gen_idx = int(args.pop_file[-5])
    return population, gen_idx


def calc_Q(args, which_states='init'):
    """validation func. check Q values of ensembles"""
    global TRAIN_DATA
    TRAIN_DATA = {"traces": pickle_load(join(args.train_data_dir,
                                             'Traces.pkl')),
                  "states": pickle_load(join(args.train_data_dir,
                                             'States.pkl'))}
    global EVAL_DATA
    EVAL_DATA = {"traces": pickle_load(join(args.eval_data_dir, 'Traces.pkl')),
                 "states": pickle_load(join(args.eval_data_dir, 'States.pkl'))}
    global EVAL_STATE_ACTION_PAIRS
    EVAL_STATE_ACTION_PAIRS = create_state_action_pairs(EVAL_DATA["traces"],
                                                        EVAL_DATA["states"],
                                                        which_states)
    global REF_ENSEMBLE
    REF_ENSEMBLE = Ensemble(args.ensemble_size, args.gamma, args.batch_size,
                            args.lr, args.input_dims, args.n_actions)
    load_ensemble_nets(REF_ENSEMBLE, args.ref_ensemble_dir)
    population = generate_initial_population(args)
    gen_idx = 0
    # for generation in range(gen_idx, 1)#args.n_generations):
    #     print(f"{datetime.now()} Generation {generation}")
    mean_q_values = []
    for i, individual in enumerate(population[:20]):
        if individual.fitness_score == 0:
            mean_q_values.append(individual.validation_q_value(args, which_states))
            print(f"{datetime.now()}  {i}")
    c = torch.concat([x[0] for x in mean_q_values])
    e = torch.concat([x[1] for x in mean_q_values])
    # population = evolve(population)
    pd.DataFrame({'cand_mean': c.detach().numpy(), 'eval_mean': e.detach().numpy()}).to_csv('ensembles_mean_q_values_2.csv')

def main(args, which_states='init'):
    global TRAIN_DATA
    TRAIN_DATA = {"traces": pickle_load(join(args.train_data_dir,
                                             'Traces.pkl')),
                  "states": pickle_load(join(args.train_data_dir,
                                             'States.pkl'))}
    global EVAL_DATA
    EVAL_DATA = {"traces": pickle_load(join(args.eval_data_dir, 'Traces.pkl')),
                 "states": pickle_load(join(args.eval_data_dir, 'States.pkl'))}
    global EVAL_STATE_ACTION_PAIRS
    EVAL_STATE_ACTION_PAIRS = create_state_action_pairs(EVAL_DATA["traces"],
                                                        EVAL_DATA["states"],
                                                        which_states)
    global REF_ENSEMBLE
    REF_ENSEMBLE = Ensemble(args.ensemble_size, args.gamma, args.batch_size,
                            args.lr, args.input_dims, args.n_actions)
    load_ensemble_nets(REF_ENSEMBLE, args.ref_ensemble_dir)

    # start a new wandb run to track this script
    args_dict = {}
    for arg_name, arg_value in vars(args).items():
        args_dict[arg_name] = arg_value
    wandb.init(
        # set the wandb project where this run will be logged
        project="genetic_algorithm_process",
        # track hyper parameters and run metadata
        config=args_dict
    )

    # threading
    num_threads = 3
    population_ = []
    # Define a function to be run on each input

    def process_input(individual):
        # Do some work on the input...
        if individual.fitness_score == 0:
            individual.calc_fitness(args)
        population_.append(individual)


    # Define a function to run in each thread
    def thread_worker(inputs, gen_id, thread_num):
        while inputs:
            input = inputs.pop(0)
            # print(datetime.now(), thread_num, input.chromosome)
            process_input(input)
            print(f"{datetime.now()} generation {gen_id} individual {len(population_)}")
            # log metrics to wandb
            wandb.log({"generation": gen_id,
                       "fitness_score": input.fitness_score})

    # genetic algorithm procedure
    if args.pop_file:
        population, gen_idx = load_population(args)
    else:
        population = generate_initial_population(args)
        gen_idx = 0

    for generation in range(gen_idx, args.n_generations):
        print(f"{datetime.now()} Generation {generation}")

        # use threads
        inputs = population
        # Create a list of threads
        threads = [threading.Thread(target=thread_worker,
                                    args=(inputs, generation, i))
                   for i in range(num_threads)]

        # Start all the threads
        for thread in threads:
            thread.start()

        # Wait for all the threads to finish
        for thread in threads:
            thread.join()

        population = population_
        population_ = []
        # end of threads code

        # # original. no threads
        # for i, individual in enumerate(population):
        #     if individual.fitness_score == 0:
        #         individual.calc_fitness(args)
        #     print(f"{datetime.now()} generation: {generation}, individual: {i}")

        save_population(population, generation)
        population = evolve(population)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='policy evaluation')
    parser.add_argument('--gamma', help='discount factor', type=float)
    parser.add_argument('--lr', help='learning rate', type=float)
    parser.add_argument('--n_actions', help='number of actions', type=int)
    parser.add_argument('--batch_size', help='batch size', type=int)
    parser.add_argument('--input_dims', help='input dimensions', type=list)
    parser.add_argument('--ensemble_size', help='number of nets in ensemble',
                        type=float)
    parser.add_argument('--train_rounds', help='number of train rounds',
                        type=int)
    parser.add_argument('--chromosome_size',
                        help='number of traces in a summary',
                        type=int)
    parser.add_argument('--population_size',
                        help='number of individuals in a single generation',
                        type=int)
    parser.add_argument('--env', help='environment to train in',
                        type=str, default='highway-v0')
    parser.add_argument('--train_data_dir', help='path to train data',
                        type=str)
    parser.add_argument('--output_dir', help='path to save results in',
                        type=str)
    parser.add_argument('--eval_data_dir', help='path to eval data',
                        type=str)
    parser.add_argument('--ref_ensemble_dir',
                        help='path to saved nets of reference ensemble',
                        type=str)
    parser.add_argument('--pop_file', help='path to saved population file',
                        type=str, default=None)
    parser.add_argument('--mutation_rate',
                        help='mutation rate for genetic algorithm',
                        type=float, default=0.05)
    parser.add_argument('--fitness_func',
                        help='score function to calculate fitness score')
    parser.add_argument('--which_state',
                        help='which state to use when computing scores. '
                             'options {"init", "rand"}', type=str)
    args = parser.parse_args()

    args.output_dir = abspath(
        join('..', "collected_data/results/new_different_agent",
             "genetic_algorithm_results", "chromosome_size_7", "score_2_mean", "rand_state_100"))

    if not exists(args.output_dir):
        makedirs(args.output_dir)

    # to use none default env uncomment and set value of the line below
    args.env = "highway2-v0"
    args.gamma = 0.99
    args.lr = 0.001
    args.batch_size = 32
    args.ensemble_size = 5
    args.train_rounds = 5000
    args.n_candidates = 1000
    args.chromosome_size = 7
    args.n_generations = 6
    args.population_size = 500
    args.fitness_func = calc_score_2_mean  # possible values: {calc_score_2, calc_score_2_mean, calc_score_5, "calc_score_5_mean", "reward_sum"}
    args.which_state = 'rand'

    env = gym.make(args.env)

    args.input_dims = [math.prod(env.observation_space.shape)]
    args.n_actions = env.action_space.n

    args.train_data_dir = abspath(
        join("..",
             "collected_data/results/new_different_agent/train_data"))
    args.eval_data_dir = abspath(
        join("..", "collected_data/results/new_different_agent/eval_data_100_traces"))
    args.ref_ensemble_dir = abspath(join(
        "..", "collected_data/results/new_different_agent/train_data/eval ensemble"))

    # to load saved population uncomment this
    # args.pop_file = abspath(
    #     join("..",
    #          "collected_data/episodes_with_random_actions/new_default_agent/genetic_algorithm_results/chromosome_size_7/score_5_mean/init_state_500/_gen_0.csv"))

    save_args(args)
    main(args, which_states=args.which_state)
    # calc_Q(args, which_states='all')
    # pop = load_population(args)
    print("DONE!!!")
