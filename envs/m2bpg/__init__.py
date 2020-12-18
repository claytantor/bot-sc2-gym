import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

from games.m2b2 import MoveToBeacon

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

SURFACE_ID = 1
START_ID = 3
GOAL_ID = 2

def gen_randpair(size_x, size_y):
    rand_x = np.random.randint(0, size_x)
    rand_y = np.random.randint(0, size_y)
    return rand_x, rand_y

def generate_map_new(size=8):
    rows = np.array(np.ones(size, dtype=np.int64))
    for row_i in range(size-1):
        newrow = np.array(np.ones(size, dtype=np.int64))
        rows = np.vstack([rows, newrow])

    rand_x_start, rand_y_start = gen_randpair(rows.shape[0],rows.shape[1])
    rows[rand_x_start][rand_y_start] = START_ID # start
    
    rand_x_goal, rand_y_goal = gen_randpair(rows.shape[0],rows.shape[1])

    # dont overwrite the start with goal
    while rand_x_start == rand_x_goal and rand_y_start == rand_y_goal:
        rand_x_goal, rand_y_goal = gen_randpair(rows.shape[0],rows.shape[1])
        if rand_x_start != rand_x_goal and rand_y_start != rand_y_goal:
            rows[rand_x_goal][rand_y_goal] = GOAL_ID # goal
    
    rows[rand_x_goal][rand_y_goal] = GOAL_ID
    
    return rows

def to_s(row, col, ncol):
    return row*ncol + col

def inc(row, col, a, nrow, ncol):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    return (row, col)

def update_probability_matrix(row, col, action, desc, nrow, ncol):
    newrow, newcol = inc(row, col, action, nrow, ncol)
    newstate = to_s(newrow, newcol, ncol)
    newval = desc[newrow, newcol]
    
    done = newval in [GOAL_ID]

    reward = float(newval in [GOAL_ID])
    return newstate, reward, done


class MoveToBeaconPygameEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, grid_size=8):

        self.grid_size = grid_size
        desc = generate_map_new(size=grid_size)
        self.desc = desc = np.asarray(desc, dtype='int16')
        # self.nrow, self.ncol = nrow, ncol = desc.shape

        self.game = MoveToBeacon(map_info=desc)

        nS, nA, P, isd = self._init_non_game(grid_size)    
        super(MoveToBeaconPygameEnv, self).__init__(nS, nA, P, isd)

    def _init_non_game(self, grid_size):
        self.grid_size = grid_size
        desc = generate_map_new(size=grid_size)
        self.desc = desc = np.asarray(desc, dtype='int16')

        self.game.set_map(desc)
        self.game.init()

        self.nrow, self.ncol = nrow, ncol = desc.shape

        nA = 4
        nS = nrow * ncol
        self.reward_range = (0, 1)
        #isd = np.array(desc == b'S').astype('float64').ravel()
        # find the start == 3
        np.seterr(divide='ignore', invalid='ignore')
        isd = np.array(desc == START_ID).astype('float64').ravel()
        isd /= isd.sum()
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col, ncol)
                for a in range(4):
                    li = P[s][a]
                    map_val = desc[row, col]

                    if map_val in [GOAL_ID]:
                        li.append((1.0, s, 0, True))
                    else:
                        li.append((1., *update_probability_matrix(row, col, a, self.desc, nrow, ncol)))

        return nS, nA, P, isd

    def reset(self):
        nS, nA, P, isd = self._init_non_game(grid_size=self.grid_size) # generate new map  
        super(MoveToBeaconPygameEnv, self).__init__(nS, nA, P, isd)
        return super(MoveToBeaconPygameEnv, self).reset()

    def step(self, action):
        # step
        self.game.step(action)

        state_1, reward_val, episode_done, p = super(MoveToBeaconPygameEnv, self).step(action)

        if self.game.game_over():
            # print("ran out of steps", self.game.tick_count, reward_val)
            episode_done = True

        return (state_1, reward_val, episode_done, p )


    def render(self, mode='human'):

        if mode in ['ansi','human']:
            # mode = 'human'

            outfile = StringIO() if mode == 'ansi' else sys.stdout

            row, col = self.s // self.ncol, self.s % self.ncol
            desc = self.desc.tolist()
            desc = [[c.decode('utf-8') for c in line] for line in desc]
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
            if self.lastaction is not None:
                outfile.write("  ({})\n".format(
                    ["Left", "Down", "Right", "Up"][self.lastaction]))
            else:
                outfile.write("\n")
            outfile.write("\n".join(''.join(line) for line in desc)+"\n")

            if mode != 'human':
                with closing(outfile):
                    return outfile.getvalue()