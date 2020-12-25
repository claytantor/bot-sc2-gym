import sys, os
import logging

from itertools import count
import uuid
import argparse
import time
import sys
import os

from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from numpy.lib.twodim_base import tril_indices_from
from plot import plot_scores, show_screen, moving_average
from envcollect import get_pretty_env_info
 
from agents.dqn3 import TerranAgent
from envs.pysc2 import MoveToBeaconPySC2Env

from utils import logit, writeline

tid = str(uuid.uuid4()).replace('-','')[:8]   

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

success = []
episodes_steps_list = []
episode_scores = []

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='logs/{}.log'.format("sc2dnn"), level=logging.DEBUG)

def main(argv):

    load_dotenv(find_dotenv())

    logit(get_pretty_env_info())

    print("loading starcraft from {}".format(os.getenv("SC2PATH")))

    # Read in command-line parameters
    parser = argparse.ArgumentParser()

    # parser.add_argument("-e", "--env", action="store", default="FrozenLake-v0", dest="env", help="environment")

    parser.add_argument("-i", "--episodes", action="store", default=5000, type=int, dest="episodes", help="episodes")

    # parser.add_argument("-a", "--agent", action="store", default="dqn.Agent", dest="agent", help="agent")

    args = parser.parse_args()
    logit("EPISODES: {}".format(args.episodes))

    env = MoveToBeaconPySC2Env()
    agent = TerranAgent(env)
    agent.reset()
    
    try:
        num_episodes = args.episodes
        for i_episode in range(num_episodes):
            logit("i_episode: {}".format(i_episode))
            # Initialize the environment and state
            timesteps = env.reset()
            last_state = env.get_state()
            current_state = env.get_state()
            state = current_state - last_state
            done = timesteps[0].last()

            for t in count():

                logit("A1")
                step_action_tensor, step_action_pysc2 = agent.step(timesteps[0])
                logit("A2")
                # time.sleep(0.02)

                # do the mapped action
                done, reward_tensor, timesteps = env.step([step_action_pysc2])

                logit("A3")
                # time.sleep(0.02)
                if reward_tensor.cpu().detach().numpy()[0] > 0.0:
                    logit("==== goal.")
                    success.append(1)
                else:
                    success.append(0)
                logit("B")
                # time.sleep(0.02)

                # Observe new state
                last_state = current_state
                current_state = env.get_state()
                logit("C")
                # time.sleep(0.02) 

                if not done:
                    next_state = current_state - last_state
                else:
                    next_state = None
                    score = timesteps[0].observation.score_cumulative['score']
                    if score == 0:
                        # negative reward done without score
                        reward_tensor = env.punish_value() 

                # Store the transition in memory
                logit("D")
                time.sleep(0.02)
                agent.push(state, step_action_tensor, next_state, reward_tensor)
                logit("E")
                time.sleep(0.02)
                if done:
                    break    

                state = next_state

                logit("F")
                time.sleep(0.02)
            
            if i_episode % 10 == 0 and i_episode != 0:
                logit("M 1")
                success_val = sum(success[-1000:])
                logit("===== success rate:{} episode:{}".format(success_val, i_episode))
                episode_scores.append(float(success_val)) 
                writeline('{},{}'.format(success_val, i_episode),'/workspace/train-sc2-{}.txt'.format(tid))
                logit("M 2")
                # plot_scores(episode_scores, "success rate/1K steps")
            
            logit("G")

    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
    main(sys.argv[1:])
