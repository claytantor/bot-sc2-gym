import sys, os
import numpy as np
from absl import flags
from io import StringIO
from pathlib import Path
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS

from agents.terran import TerranAgent

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


no_op=0
move_camera=1
select_point=2
select_rect=3
select_control_group=4
select_idle_worker=6
select_army=7
Attack_screen=12
Attack_minimap=13
Build_Barracks_screen=42
Build_CommandCenter_screen=44
Build_EngineeringBay_screen=50
Build_SupplyDepot_screen=91
Effect_CalldownMULE_screen=183
Effect_Stim_quick=234
Morph_OrbitalCommand_quick=309
Move_screen=331
Move_minimap=332
Patrol_screen=333
Patrol_minimap=334
Smart_screen=451
Smart_minimap=452
Train_SCV_quick=490


GRID_SIZE=16

# make grid move actions
def make_move_actions_old(window_width, window_height):
    actions_model = {}
    grid_units_w = int(window_width/GRID_SIZE)
    grid_units_h = int(window_height/GRID_SIZE)
    tensor_index = 0
    for h_inc in range(0,grid_units_w): #rows
        for w_inc in range(0,grid_units_h):
            action_id = 3000+(h_inc*GRID_SIZE)+w_inc
            actions_model[action_id] = {
                'tensor_index': tensor_index,
                'syc2_fid':Move_screen,
                'dscr':'move screen location w{}_h{}'.format(w_inc, h_inc),
                'location':(w_inc, h_inc),
                'loc_index':(w_inc*GRID_SIZE, h_inc*GRID_SIZE),
                'pysc2_action': actions.FUNCTIONS.Move_screen("now", (w_inc*GRID_SIZE, h_inc*GRID_SIZE))
            }
            tensor_index += 1

    return actions_model

def make_move_actions(window_width, window_height):
    actions_list = []
    grid_units_w = int(window_width/GRID_SIZE)
    grid_units_h = int(window_height/GRID_SIZE)
    tensor_index = 0
    for h_inc in range(0,grid_units_w): #rows
        for w_inc in range(0,grid_units_h):
            action_id = 3000+(h_inc*GRID_SIZE)+w_inc
            actions_model = {
                'action_id': action_id,
                'tensor_index': tensor_index,
                'syc2_fid':Move_screen,
                'dscr':'move screen location w{}_h{}'.format(w_inc, h_inc),
                'location':(w_inc, h_inc),
                'loc_index':(w_inc*GRID_SIZE, h_inc*GRID_SIZE),
                'pysc2_action': actions.FUNCTIONS.Move_screen("now", (w_inc*GRID_SIZE, h_inc*GRID_SIZE))
            }
            actions_list.append(actions_model)
            tensor_index += 1

    return actions_list


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

class MoveToBeaconPySC2Env():
    def __init__(self,         
        map_name='MoveToBeacon',
        render=False,
        reset_done=True,
        max_ep_len=None,
        spatial_dim=16,
        screen=96,
        minimap=64,
        step_mul=8,
        obs_features=None):


        # fail-safe if executed not as absl app
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(sys.argv)
  
        replaysDir = os.path.join(Path(__file__).parent.absolute(),'replays')

        self.screen = screen
        self.minimap = minimap

        self._sc2env = sc2_env.SC2Env(
                    map_name="MoveToBeacon",
                    players=[sc2_env.Agent(sc2_env.Race.terran, "Tergot"),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(
                            screen=self.screen, minimap=self.minimap),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    realtime=True,
                    save_replay_episodes=1,
                    replay_dir=replaysDir,
                    visualize=False) 
        
        self.action_model = make_move_actions(self.screen, self.screen)

        self.timesteps = self._sc2env.reset()
  

    def reset(self):
        timesteps = self._sc2env.reset()
        obs = timesteps[0]
        self.state = self.make_state(obs)
        return timesteps

    def get_actions(self):
        return self.action_model

    def observation_spec(self):
        return self._sc2env.observation_spec()
    
    def action_spec(self):
        return self._sc2env.action_spec()

    def step(self, actions):
        timesteps = self._sc2env.step(actions)
        obs = timesteps[0]
        # visibility_screen = obs.observation['feature_screen'][1]

        # zeros for now but should be a 3d array of 
        # screen = np.zeros((3, self.screen, self.screen))
        # screen = obs.observation['feature_screen']
        # screen = np.ascontiguousarray(screen, dtype=np.float32) / 1.0
        # screen = torch.from_numpy(screen).to(device)
        self.state = self.make_state(obs)

        reward_tensor = torch.tensor(np.array([obs.reward]), dtype=torch.float64, device=device)

        return obs.last(), reward_tensor, timesteps

    def make_state(self, obs, name='feature_screen', dimension_name='visibility_map'):
        screen = obs.observation[name][dimension_name]
        screen = np.ascontiguousarray(screen, dtype=np.float32) / np.amax(screen)
        
        # screen = torch.from_numpy(screen)
        # return resize(screen).unsqueeze(0).to(device)

        screen = np.array([screen])
        screen = torch.from_numpy(screen).to(device)
        return screen


    def get_state(self):
        return self.state



