import sys, os
import numpy as np

from io import StringIO
from pathlib import Path
from contextlib import closing
from gym import utils
from gym.envs.toy_text import discrete

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

class MoveToBeaconPySC2Env(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'rgb']}

    def __init__(self, grid_size=8):

        replaysDir = os.path.join(Path(__file__).parent.absolute(),'replays')

        # self.grid_size = grid_size
        # desc = generate_map_new(size=grid_size)
        # self.desc = desc = np.asarray(desc, dtype='int16')
        # # self.nrow, self.ncol = nrow, ncol = desc.shape

        # self.game = MoveToBeacon(map_info=desc)

        self._scenv = sc2_env.SC2Env(
                    map_name="MoveToBeacon",
                    players=[sc2_env.Agent(sc2_env.Race.terran, "Tergot"),
                             sc2_env.Bot(sc2_env.Race.random,
                                         sc2_env.Difficulty.very_easy)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(
                            screen=96, minimap=64),
                        use_feature_units=True),
                    step_mul=16,
                    game_steps_per_episode=0,
                    realtime=False,
                    save_replay_episodes=1,
                    replay_dir=replaysDir,
                    visualize=True) 


        nS, nA, P, isd = self._init_non_game(grid_size)    
        super(MoveToBeaconPySC2Env, self).__init__(nS, nA, P, isd)    

    def _init_non_game(self, grid_size):
        self.grid_size = grid_size

        nS = None
        nA = None
        P = None
        isd = None
        

        return (nS, nA, P, isd)