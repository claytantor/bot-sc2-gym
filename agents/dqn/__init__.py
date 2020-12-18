import math
import numpy as np

from gym import error, spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

# hyper parameters
EPS_START = 0.99  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
BATCH_SIZE = 64  # Q-learning batch size

# two types of observation spaces
# gym.spaces.discrete.Discrete and gym.spaces.box.Box
# its easy to add.
def get_space_size(space):
    if type(space) is spaces.discrete.Discrete:
        print('gym.spaces.discrete.Discrete')
        return space.n
    else:
        raise ValueError("Unknown space type.")

class Agent(nn.Module):
    def __init__(self, env, optimizer_type='Adam'):
        super(Agent, self).__init__()

        self.observation_space_size = get_space_size(env.observation_space)
        self.hidden_size = get_space_size(env.observation_space)
        self.action_space_size = get_space_size(env.action_space)

        self.action_space = env.action_space
        self.optimizer_type = optimizer_type

        self.learning_rate = 0.001
        self.momentum = 0.9
        self.gamma = 0.9
        self.lr_step=100

        self.steps_done = 0

        self.l1 = nn.Linear(in_features=self.observation_space_size, out_features=self.hidden_size)
        self.l2 = nn.Linear(in_features=self.hidden_size, out_features=self.action_space.n)
        self.uniform_linear_layer(self.l1.to(device=device))
        self.uniform_linear_layer(self.l2.to(device=device))

        # optimizer
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
   
    def forward(self, state):
        obs_emb = self.one_hot([int(state)], self.observation_space_size)
        out1 = torch.sigmoid(self.l1(obs_emb))
        return self.l2(out1).view((-1))

    def uniform_linear_layer(self, linear_layer):
        linear_layer.weight.data.uniform_()
        linear_layer.bias.data.fill_(-0.02)

    def one_hot(self, ids, nb_digits):
        """
        ids: (list, ndarray) shape:[batch_size]
        """
        if not isinstance(ids, (list, np.ndarray)):
            raise ValueError("ids must be 1-D list or array")
        batch_size = len(ids)
        ids = LongTensor(ids).view(batch_size, 1)
        out_tensor = Variable(FloatTensor(batch_size, nb_digits))
        out_tensor.data.zero_()
        out_tensor.data.scatter_(dim=1, index=ids, value=1.)
        return out_tensor

    def choose_action(self, s):
        if (np.random.rand(1) < 0.1): 
            return self.action_space.sample()
        else:
            agent_out = self(s).detach()
            _, max_index = torch.max(agent_out, 0)
            return max_index.data.cpu().numpy().tolist()
     
    def update(self, agent_action, last_state, step_state, reward, done, probability):
        # calculate target and loss
        target_q = reward + 0.99 * torch.max(self(step_state).detach()) # detach from the computing flow
        loss = F.smooth_l1_loss(self(last_state)[agent_action], target_q)
        
        # update model to optimize Q
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()