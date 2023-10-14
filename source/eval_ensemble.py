from os.path import abspath, join
from datetime import datetime
import torch as T
import numpy as np
import wandb

from agent import Agent
from utils import compute_cumulative_discounted_reward


class EvalNet(Agent):
    def __init__(self, gamma: float, lr: float, n_actions: int, batch_size: int,
                 input_dims):
        super().__init__(gamma, epsilon=0, lr=lr, input_dims=input_dims,
                         batch_size=batch_size, n_actions=n_actions)

        self.train_loss = []
        self.q_value_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.next_q_value_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.next_action_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_replay_buffer(self, traces, states):
        """store learning data"""
        for trace in traces:
            for i in range(trace.length):
                index = self.mem_cntr % self.mem_size # throw error
                self.state_memory[index] = trace.obs[i]
                self.new_state_memory[index] = trace.new_obs[i]
                self.reward_memory[index] = trace.rewards[i]
                self.action_memory[index] = trace.actions[i]
                self.terminal_memory[index] = trace.dones[i]
                self.q_value_memory[index] = states[trace.states[i]].\
                    observed_actions[trace.actions[i]]
                self.next_action_memory[index] = T.argmax(
                    states[(trace.states[0][0], i+1)].observed_actions)
                if i + 1 < trace.length:
                    self.next_q_value_memory[index] = states[trace.states[i+1]]\
                        .observed_actions[trace.actions[i+1]]
                else:
                    self.next_q_value_memory[index] = 0
                self.mem_cntr += 1

        # print("NUM OF POINTS: ", self.mem_cntr)

    def learn(self):
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]). \
            to(self.Q_eval.device)
        next_action_batch = T.tensor(self.next_action_memory[batch]). \
            to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        # terminal_batch = T.tensor(self.terminal_memory[batch]). \
        #     to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        """here I take the max q value of the next state using dqn of the 
        original agent
        """
        # q_values_next = T.tensor(self.next_q_value_memory[batch]).\
        #     to(self.Q_eval.device)

        """here I evaluate the q values of the next state using the current dqn
        and take the action that was taken by the original agent
        at this state 
        """
        q_values_next = self.Q_eval.forward(new_state_batch).\
            gather(1, next_action_batch.type(T.int64).view(-1, 1)).view(-1)
        # x = self.Q_eval.forward(new_state_batch)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_target = reward_batch + self.gamma * q_values_next

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.train_loss.append(round(loss.data.item(), 7))


class Ensemble:
    def __init__(self, ensemble_size: int, gamma: float, batch_size: int,
                 learning_rate: float, input_dims, n_actions):

        self.size = ensemble_size
        self.eval_nets = [
            EvalNet(gamma, learning_rate, n_actions, batch_size, input_dims)
            for _ in range(self.size)
        ]

    def get_eval_nets(self):
        """:return: list of evaluation dqn"""
        return [eval_net.Q_eval for eval_net in self.eval_nets]

    def load_train_data(self, traces, states):
        """load replay buffer to each net"""
        for eval_net in self.eval_nets:
            eval_net.store_replay_buffer(traces, states)

    def train_nets(self, train_rounds, output_dir, save=True):
        """only run after load_train_data run"""

        # log to wandb
        # args_dict = {}
        # for arg_name, arg_value in vars(args).items():
        #     args_dict[arg_name] = arg_value

        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project="ensemble train",
        #     # track hyper parameters and run metadata
        #     config=args_dict
        # )

        for j in range(self.size):
            # print(self.eval_nets[j].Q_eval.device)
            for i in range(train_rounds):
                self.eval_nets[j].learn()

                # wandb.log({"net number": j, "train round": i,
                #            "loss": self.eval_nets[j].train_loss[-1]})
                # if (i + 1) % 100 == 0:
                #     print(f"{datetime.now()} Net {j + 1}, Train round {i + 1}, avg loss: "
                #           f"{np.mean(self.eval_nets[j].train_loss[-100:])}")
            if save:  # save a pkl file of nets
                T.save(self.eval_nets[j].Q_eval.state_dict(),
                       join(abspath(output_dir), f'net_{j}.pkl'))


def calc_score_1(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:
    """
    calculate score of the candidate ensemble relative to
    the evaluation ensemble, based on given traces.
    :param candidate_ensemble:
    :param eval_ensemble:
    :param states_actions: list of state-action tuples.
    :return: float.
    """
    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions])
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_variance = T.var(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_variance = T.var(eval_q_values, dim=0)

    return (T.sum(cand_variance) / T.sum(eval_variance)).item()


def calc_score_2(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:

    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions]).to()
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_variance = T.var(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_variance = T.var(eval_q_values, dim=0)

    return T.sum((cand_variance / eval_variance)).item()


def calc_score_2_mean(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:

    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions]).to()
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_variance = T.var(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_variance = T.var(eval_q_values, dim=0)

    return T.sum((cand_variance / eval_variance)).item() / len(states_actions)


def calc_score_3(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:

    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions])
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_variance = T.var(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_variance = T.var(eval_q_values, dim=0)

    s = cand_variance - eval_variance
    s[s < 0] = 0

    return T.sum(s).item()


def calc_score_4(eval_ensemble: Ensemble, states_actions,
                 discounted_rewards: list) -> float:
    """

    :param candidate_ensemble:
    :param eval_ensemble:
    :param states_actions: list of state_action tuples which was created
    using cand_traces
    :param discounted_rewards: list of discounted rewards of traces which
    were used for training candidate_ensemble
    :return:
    """
    if len(discounted_rewards) != len(states_actions):
        raise ValueError("params discounted_rewards and states_actions "
                         "should have the same length")
    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions])
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_mean = T.mean(eval_q_values, dim=0)

    return T.sum(eval_mean - T.tensor(discounted_rewards)).item()


def calc_score_5(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:

    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions])
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_mean = T.mean(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_mean = T.mean(eval_q_values, dim=0)

    return T.sum(T.pow(eval_mean - cand_mean, 2)).item()


def calc_score_5_mean(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:
    """like score_5 but taking the mean of the sum"""
    state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions])
    action_batch = np.array([s_a[1] for s_a in states_actions])
    batch_index = np.arange(len(states_actions), dtype=np.int32)

    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_mean = T.mean(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_mean = T.mean(eval_q_values, dim=0)

    return T.sum(T.pow(eval_mean - cand_mean, 2)).item() / len(states_actions)


def calc_q_values(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions, init_only=True) -> tuple:
    if init_only:
        state_batch = T.stack([T.from_numpy(s_a[0]) for s_a in states_actions])
        action_batch = np.array([s_a[1] for s_a in states_actions])
        batch_index = np.arange(len(states_actions), dtype=np.int32)
    else:
        batch_size = 50
        indices = np.random.choice(np.arange(len(states_actions), dtype=np.int32),
                                   batch_size, replace=False)
        state_batch = T.stack([T.from_numpy(states_actions[i][0]) for i in indices])
        action_batch = np.array([states_actions[i][1] for i in indices])
        batch_index = np.arange(batch_size, dtype=np.int32)


    cand_nets = candidate_ensemble.get_eval_nets()
    cand_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in cand_nets])

    cand_mean = T.mean(cand_q_values, dim=0)

    eval_nets = eval_ensemble.get_eval_nets()
    eval_q_values = T.stack([dqn.forward(state_batch)[batch_index, action_batch]
                             for dqn in eval_nets])

    eval_mean = T.mean(eval_q_values, dim=0)
    return cand_mean, eval_mean

    # return T.sum(T.pow(eval_mean - cand_mean, 2)).item()


def calc_score_6_validation(candidate_ensemble: Ensemble, eval_ensemble: Ensemble,
                 states_actions) -> float:
    """sum of rewards"""
    return 0.0
