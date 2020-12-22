import math
import numpy as np

from gym import error, spaces

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import sys
import os
from dotenv import load_dotenv, find_dotenv


from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

import random

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN2(nn.Module):



    def __init__(self):
        super(DQN2, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out

class DQN(nn.Module):
# Layer	Input	    kernal 	    Stride	# filters	Activa	Output
# conv1	96x96x1	    8x8	        4	    32	        ReLU	20x20x32
# conv2	20x20x32	4x4	        2	    64	        ReLU	9x9x64
# conv3	9x9x64	    3x3	        1	    64	        ReLU	7x7x64=3136
# fc4	7x7x64		                    512	        ReLU	512
# fc5	512			                    2	        Linear	2
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=8, stride=4) 
            # output = torch.Size([1, 32, 23, 23])
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2) 
            # output = torch.Size([1, 64, 10, 10])
        self.bn2 = nn.BatchNorm2d(64)

        # self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1) 
            # output = torch.Size([1, 64, 8, 8])
        self.bn3 = nn.BatchNorm2d(64)

        # # Number of Linear input connections depends on output of conv2d layers
        # # and therefore the input size, so compute it.
        # def conv2d_size_out(size, kernel_size = 3, stride = 1):
        #     return (size - (kernel_size - 1) - 1) // stride  + 1

        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        # linear_input_size = convw * convh * 32
        # self.head = nn.Linear(linear_input_size, outputs)


        self.fc4 = nn.Linear(4096, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, outputs)



    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # return self.head(x.view(x.size(0), -1))

        x = x.view(x.size()[0], -1)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        return x


class TerranAgent(base_agent.BaseAgent):
    def __init__(self, env, optimizer_type = 'Adam'):
        super(TerranAgent, self).__init__()

        self._env = env
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.gamma = 0.9
        self.lr_step=100
        
        self.actions = env.get_actions()
        self.n_actions = len(self.actions)

        _, screen_height, screen_width =env.observation_spec()[0].feature_screen

        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        elif optimizer_type == 'Nesterov':
            self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=self.learning_rate, momentum=self.momentum, nesterov=True)
        else:
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)

        self.memory = ReplayMemory(10000)

        self.setup(env.observation_spec(), env.action_spec())

        # do not reset
        self.steps_done = 0


    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def push(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        self.optimize_model()


    # def make_state(self, obs, name='feature_screen'):
    #     screen = obs.observation[name]
    #     screen = np.ascontiguousarray(screen, dtype=np.float32) / np.amax(screen)
    #     return torch.from_numpy(screen).to(device)

    def step(self, obs):
        """
        use the observation and return an action
        """
        super(TerranAgent, self).step(obs)

        state = self._env.make_state(obs)

        tensor_action = self.select_action_tensor(state)
        # print("tensor_action",tensor_action)

        #lookup the pysc2 action that maps to the tensor
        action_index = tensor_action.data.cpu().numpy()[0][0]

        action_model = list(filter(lambda x: x['tensor_index'] == action_index, self.actions))

        # this only work for m2b
        if not self.unit_type_is_selected(obs, units.Terran.Marine):
            marines = self.get_units_by_type(obs, units.Terran.Marine)
            # select one
            marine = random.choice(marines)
            return tensor_action, actions.FUNCTIONS.select_point("select_all_type", (marine.x, marine.y))
        else:
            return tensor_action, action_model[0]['pysc2_action']



    def select_action_tensor(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)



    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())
        # criterion = nn.MSELoss()
        # loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    
    def first(self, obs):
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()

            if xmean <= 31 and ymean <= 31:
                self.attack_coordinates = (49, 49)
            else:
                self.attack_coordinates = (12, 16)
    
    def build_buildings(self, obs):
        supply_depot = self.get_units_by_type(obs, units.Terran.SupplyDepot)
        
        if len(supply_depot) == 0:
            if self.unit_type_is_selected(obs, units.Terran.SCV):
                if self.can_do(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_SupplyDepot_screen("now", (x, y)) 
            

            scvs = self.get_units_by_type(obs, units.Terran.SCV)
            if len(scvs) > 0:
                scv = random.choice(scvs)

                return actions.FUNCTIONS.select_point("select_all_type", (scv.x,
                                                                          scv.y))

        else:
            barracks = self.get_units_by_type(obs, units.Terran.Barracks)   
            if len(barracks) == 0:
                if self.can_do(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                    x = random.randint(0, 83)
                    y = random.randint(0, 83)

                    return actions.FUNCTIONS.Build_Barracks_screen("now", (x, y)) 

        return None

    def build_units(self, obs):
        barracks = self.get_units_by_type(obs, units.Terran.Barracks)
        marines = self.get_units_by_type(obs, units.Terran.Marine)

        if len(barracks) > 0 and len(marines) < 11:
            if self.unit_type_is_selected(obs, units.Terran.Barracks):
                if self.can_do(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                    return actions.FUNCTIONS.Train_Marine_quick("now")
            else:
                barrack = random.choice(barracks)
                return actions.FUNCTIONS.select_point("select_all_type", (barrack.x,
                                                                          barrack.y))
      
        return None


    def attack(self, obs):

        marines = self.get_units_by_type(obs, units.Terran.Marine)
        if len(marines) >= 10:
            if self.unit_type_is_selected(obs, units.Terran.Marine):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now",
                                                            self.attack_coordinates)

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")   

        return None