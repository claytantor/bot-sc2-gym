import sys, os
import numpy as np
import uuid

from absl import flags
from pathlib import Path

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from utils import logit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

pickleDir = os.path.join('/workspace','pickle')

device = torch.device("cpu")
if int(os.getenv("USE_CUDA"))==1:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.backends.cudnn.enabled = False
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
logit("ENV USING {} DEVICE".format(str(device)))

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

def make_noop(tensor_index):
    return {
                'tensor_index': tensor_index,
                'pysc2_action': actions.FUNCTIONS.no_op()
            }

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

    # actions_list.append(make_noop(tensor_index))

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
        obs_features=None,
        realtime=False,
        visualize=False):


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
                    realtime=realtime,
                    save_replay_episodes=0,
                    replay_dir=replaysDir,
                    visualize=visualize) 
        
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

    def punish_value(self):
        return torch.tensor(np.array([-1]), dtype=torch.float64, device=device)

    def step(self, actions):
        timesteps = self._sc2env.step(actions)
        obs = timesteps[0]
        self.state = self.make_state(obs)
        
        # logit("step 1 before reward_tensor")
        reward_tensor = torch.tensor(np.array([obs.reward]), dtype=torch.float64, device=device)
        # logit("step 2 after reward_tensor")

        return obs.last(), reward_tensor, timesteps

    def make_state(self, obs, name='feature_screen', dimension_name='player_relative'):

        # tid = str(uuid.uuid4()).replace('-','')[:8]        

        # logit("make_state start {}".format(tid))
        screen = obs.observation[name][dimension_name]
        if np.amax(screen) > 0:
            screen = np.ascontiguousarray(screen, dtype=np.float32) / np.amax(screen)
        
        screen = np.array([screen])

        # # save 
        # pickleFile = os.path.join(pickleDir,'pysc2-{}'.format(tid))
        # np.save(pickleFile, screen, allow_pickle=True, fix_imports=True)

        # logit("make_state create device based tensor: {}".format(tid))
        screen = torch.tensor(screen, dtype=torch.float32, device=device)
        # logit("make_state ready to return: {}".format(tid))
        return screen


    def get_state(self):
        return self.state


# python -c "import torch; torch.tensor(1).to('cuda:0')"



