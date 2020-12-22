import sys, os

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import sys
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

 
from agents.dqn3 import TerranAgent
from envs.pysc2 import MoveToBeaconPySC2Env

from absl import flags

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

episode_durations = []

def main(argv):
    print("starting sc2 bot app.")

    load_dotenv(find_dotenv())

    env = MoveToBeaconPySC2Env()
    agent = TerranAgent(env)
    agent.reset()
    
    try:
        num_episodes = 50
        for i_episode in range(num_episodes):

            print("i_episode", i_episode)
            # Initialize the environment and state
            timesteps = env.reset()
            last_state = env.get_state()
            current_state = env.get_state()
            state = current_state - last_state
            done = timesteps[0].last()
            reward = timesteps[0].reward
            for t in count():
                
                step_action_tensor, step_action_pysc2 = agent.step(timesteps[0])


                if timesteps[0].last():
                    break

                # do the mapped action
                done, reward, timesteps = env.step([step_action_pysc2])

                # Observe new state
                last_state = current_state
                current_state = env.get_state()

                if not done:
                    next_state = current_state - last_state
                else:
                    next_state = None

                # Store the transition in memory
                agent.push(state, step_action_tensor, next_state, reward)

                state = next_state



    except KeyboardInterrupt:
        pass

def get_screen():
    pass

def select_action(state):
    pass

def optimize_model():
    pass

target_net = None 
policy_net = None 


def main_old(argv):   

    env=None
    
    try:

        print("starting qlearning app.")
        num_episodes = 50
        for i_episode in range(num_episodes):
            print("i_episode", i_episode)
            # Initialize the environment and state
            env.reset()
            last_screen = get_screen()
            current_screen = get_screen()
            state = current_screen - last_screen

            for t in count():
                # Select and perform an action
                action = select_action(state)
                _, reward, done, _ = env.step(action.item())
                # reward = torch.tensor([reward], device=device)

                # Observe new state
                last_screen = current_screen
                #current_screen = get_screen()
                current_screen = None
                
                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in memory
                # memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # if t % np.random.randint(1,100) == 0 and state != None:
                #     show_screen(state, "state render")

                # Perform one step of the optimization (on the target network)
                optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    # plot_durations()
                    break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
    
    except KeyboardInterrupt:
        print('exit')



if __name__ == "__main__":
    main(sys.argv[1:])
