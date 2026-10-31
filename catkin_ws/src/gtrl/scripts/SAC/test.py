#!/usr/bin/env python3

import sys
sys.path.append('/workspace/catkin_ws/src/gtrl/scripts')

import os
import time
import yaml
import argparse
import statistics
import numpy as np
from collections import deque
import torch

from SAC.DRL import SAC
from Environments.env_lab import GazeboEnv

import rospy
from geometry_msgs.msg import Twist

def evaluate(network, env, frame_stack, max_steps, linear_cmd_scale, angular_cmd_scale, max_action, eval_episodes=10, epoch=0):
    obs_list = deque(maxlen=frame_stack)
    env.collision = 0
    ep = 0
    avg_reward_list = []
    while ep < eval_episodes:
        count = 0
        obs, goal = env.reset()
        done = False
        avg_reward = 0.0

        for i in range(4):
            obs_list.append(obs)

        observation = np.concatenate((obs_list[-4], obs_list[-3], obs_list[-2], obs_list[-1]), axis=-1)

        while not done and count < max_steps:
            if count == 0:
                action = network.choose_action(np.array(observation), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
                a_in = [(action[0] + 1) * linear_cmd_scale, action[1] * angular_cmd_scale]
                last_goal = goal
                obs_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, count)        
                observation = np.concatenate((obs_, obs_, obs_, obs_), axis=-1)
                
                for i in range(4):
                    obs_list.append(obs_)           

                if done:
                    print("\n..............................................")
                    print("Bad Initialization, skip this episode.")
                    print("..............................................")
                    env.collision -= 1
                    break

                count += 1
                continue
            
            act = network.choose_action(np.array(observation), np.array(goal[:2]), evaluate=True).clip(-max_action, max_action)
            a_in = [(act[0] + 1) * linear_cmd_scale, act[1]*angular_cmd_scale]
            obs_, _, _, _, _, _, reward, done, goal, target = env.step(a_in, count)        
            avg_reward += reward
            observation = np.concatenate((obs_list[-3], obs_list[-2], obs_list[-1], obs_), axis=-1)
            obs_list.append(obs_)
            count += 1
        
        # If it was a bad initialization, it broke out. Only increment ep if it ran completely or properly hit target/collision
        if count > 0:
            ep += 1
            avg_reward_list.append(avg_reward)
            print("\n..............................................")
            print("%i Loop, Steps: %i, Avg Reward: %f, Collision No. : %i " % (ep, count, avg_reward, env.collision))
            print("..............................................")
            
    reward = statistics.mean(avg_reward_list)
    col = env.collision
    print("\n..............................................")
    print("Average Reward over %i Evaluation Episodes, Avg Reward: %f, Total Collisions: %i" % (eval_episodes, reward, col))
    print("..............................................")
    return reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Prefix of the model to load (e.g., gtrl_run2_ep1100_reward-50_seed420)")
    parser.add_argument('--episodes', type=int, default=10, help="Number of episodes to evaluate")
    args = parser.parse_args()

    # Load configuration
    with open('/workspace/catkin_ws/src/gtrl/scripts/SAC/config.yaml', "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    algo = "GoT-SAC"
    policy_type = config[algo]['actor_type']
    critic_type = config[algo]['critic_type']
    transformer_block = config[algo]['block']
    transformer_head = config[algo]['head']

    frame_stack = config['FRAME_STACK']
    max_steps = config['MAX_STEPS']
    lr_a = config['LR_A']
    lr_c = config['LR_C']
    lr_alpha = config['LR_ALPHA']
    alpha = config['ALPHA']
    auto_tune = config['AUTO_TUNE']
    tau = config['TAU']
    gamma = config['GAMMA']
    buffer_size = config['BUFFER_SIZE']
    policy_freq = config['ACTOR_FREQ']

    seed = config['SEED']
    env_name = config['ENV_NAME']
    driver = config['DRIVER']
    linear_cmd_scale = config['L_SCALE']
    angular_cmd_scale = config['A_SCALE']
    max_action = 1

    print("Initializing Gazebo Environment for evaluation...")
    master_uri = '11311'
    # same launch file as main.py / test_all.py (must match the robot topics in env_lab.py)
    env = GazeboEnv('main.launch', master_uri, 1, 1, 1)

    time.sleep(5)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    env.seed(seed)
    state, _ = env.reset()
    action_dim = 2
    physical_state_dim = 2

    print("Initializing Agent...")
    ego = SAC(action_dim, physical_state_dim, policy_type, critic_type, False, False, False, seed, lr_c, lr_a, lr_alpha, buffer_size, tau, policy_freq, gamma, alpha, block=transformer_block, head=transformer_head, automatic_entropy_tuning=auto_tune)

    print(f"Loading Model: {args.model}")
    ego.load(args.model, directory="/workspace/catkin_ws/src/gtrl/scripts/SAC/final_models")
    
    print(f"Starting Evaluation for {args.episodes} episodes...")
    evaluate(ego, env, frame_stack, max_steps, linear_cmd_scale, angular_cmd_scale, max_action, eval_episodes=args.episodes)
