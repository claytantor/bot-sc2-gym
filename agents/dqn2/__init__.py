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


class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()

        self.attack_coordinates = None

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

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        self.first(obs)
        actions_list = []
        actions_list.append(self.build_buildings(obs))
        actions_list.append(self.build_units(obs))
        actions_list.append(self.attack(obs))

        #filter out empties
        active_actions = list(filter(lambda x: x != None, actions_list))

        if len(active_actions)==0:
            return actions.FUNCTIONS.no_op()
        else:
            return active_actions[0]
    
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