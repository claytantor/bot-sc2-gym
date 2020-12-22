import os, sys
from gym.utils.seeding import np_random
import time
import torch
import gym
import argparse

from dotenv import load_dotenv, find_dotenv

from envs.m2bpg import MoveToBeaconPygameEnv
from envs.pysc2 import MoveToBeaconPySC2Env

from agents import dqn, dqn2

from plot import plot_scores, show_screen, moving_average

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

class Trainer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.success = []
        self.episodes_steps_list = []
        self.episode_scores = []
    
    def train(self, episodes):       
        for e_i in range(episodes):
            state = self.env.reset()
            episode_done = False
            reward_val = 0
            while not episode_done:
                
                # perform chosen action
                action = self.agent.choose_action(state)

                state_1, reward_val, episode_done, p = self.env.step(action)

                if episode_done == True: 
                    if reward_val < 1: 
                        # print("FAIL...", reward_val)
                        self.success.append(0)
                        reward_val=-1
                    else:
                        # print("GOAL...", reward_val)
                        self.success.append(1)
                               
                self.agent.update(action, state, state_1, reward_val, episode_done, p)
                
                # update state
                state = state_1
                # time.sleep(0.1) #slow it down to watch
   
            if e_i % 100 == 0 and e_i != 0:
                success_percent = sum(self.success[-100:])
                print("success rate:{} episode:{}".format(success_percent, e_i))
                self.episode_scores.append(float(success_percent))
                plot_scores(self.episode_scores, "success rate per 100")

def env_factory(name):
    env = None
    print("env name:", name)
    if(name == 'MoveToBeaconPygame'):
        env = MoveToBeaconPygameEnv(grid_size=8)

    elif(name == 'MoveToBeaconPySC2Env'):
        # from absl import flags
        # FLAGS = flags.FLAGS
        # FLAGS(sys.argv)        
        env = MoveToBeaconPySC2Env()    
    else:
        env = gym.make(name)

    if env==None:
        raise ValueError("no env found")
    else:
        return env

def agent_factory(name, env):
    agent = None
    print("agent name:", name)
    if name == 'dqn.Agent':
        agent = dqn.Agent(env, optimizer_type="Adam")
    elif name == 'dqn3.Agent':
        agent = dqn2.Agent(env, optimizer_type="Adam")

    if agent==None:
        raise ValueError("no agent found")
    else:
        return agent

def main(argv): 
    # Read in command-line parameters
    # parser = argparse.ArgumentParser()

    # parser.add_argument("-e", "--env", action="store", default="FrozenLake-v0", dest="env", help="environment")

    # parser.add_argument("-i", "--episodes", action="store", default=30000, type=int, dest="episodes", help="episodes")

    # parser.add_argument("-a", "--agent", action="store", default="dqn.Agent", dest="agent", help="agent")

    # args = parser.parse_args()

    # env factory
    env = env_factory("MoveToBeaconPySC2Env")
    agent = agent_factory("dqn2.Agent", env)

    print("Action space: ", env.action_space)
    print("Observation space: ", env.observation_space)
    
    t = Trainer(agent, env)
    t.train(100)

           
if __name__ == "__main__":
    load_dotenv(find_dotenv())
    main(sys.argv[1:])
