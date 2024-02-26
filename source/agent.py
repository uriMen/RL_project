import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import copy


class DQN(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = T.device('mps:0' if T.backends.mps.is_available() else 'cpu')  # to use apple's mps
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4, repeat_train=100):
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DQN(self.lr, input_dims=input_dims, fc1_dims=256,
                          fc2_dims=256, n_actions=n_actions)
        self.Q_target = copy.deepcopy(self.Q_eval)
        self.update_target_every = 100
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.repeat_train = repeat_train
        self.losses = [0.0] * repeat_train
        self.qs = [0.0] * repeat_train


    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def act(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def get_state_action_values(self, observation):
        observation = np.array(observation).flatten()
        state = T.tensor(observation).to(self.Q_eval.device)
        return self.Q_eval.forward(state)

    def learn(self):
        # if self.mem_cntr < self.batch_size:
        if self.mem_cntr < self.mem_size:
            return
        for i in range(self.repeat_train):
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).\
                to(self.Q_eval.device)
            reward_batch = T.tensor(self.reward_memory[batch]).\
                to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).\
                to(self.Q_eval.device)
            action_batch = self.action_memory[batch]

            # Double q-learning algorithm
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0
            arg_max_ = T.argmax(q_next, dim=1)
            q_values_next = self.Q_target.forward(new_state_batch).gather(1, arg_max_.view(-1, 1)).view(-1)
            q_target = reward_batch + self.gamma * q_values_next

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            self.losses.append(float(loss))
            self.qs.append(float(q_eval.mean()))
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.Q_eval.parameters(), 1.0)
            self.Q_eval.optimizer.step()
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)

        # copy eval net to target net
        if self.mem_cntr % self.update_target_every == 0:
            # assert self.Q_eval.state_dict().__str__() != self.Q_target.state_dict().__str__()
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            # assert self.Q_eval.state_dict().__str__() == self.Q_target.state_dict().__str__()
