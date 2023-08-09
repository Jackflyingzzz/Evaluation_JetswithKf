import os
import socket
import numpy as np
import csv
import sys
import os

from Env2DCylinderModified import Env2DCylinderModified
from probe_positions import probe_positions
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
import numpy as np
from dolfin import Expression
#from printind.printind_function import printi, printiv
import math
from stable_baselines3.common.monitor import Monitor
from gym.wrappers.time_limit import TimeLimit
import os

import argparse
import os
import json
import pandas as pd
from tqdm import trange
from sb3_contrib import TQC
from stable_baselines3 import SAC
from Env2DCylinderModified import Env2DCylinderModified
from simulation_base.env import resume_env, nb_actuations, simulation_duration
from stable_baselines3.common.evaluation import evaluate_policy

# If previous evaluation results exist, delete them
if(os.path.exists("saved_models/test_strategy.csv")):
    os.remove("saved_models/test_strategy.csv")

if(os.path.exists("saved_models/test_strategy_avg.csv")):
    os.remove("saved_models/test_strategy_avg.csv")


if __name__ == '__main__':

    saver_restore = '/rds/general/user/jz1720/home/TQCPM27FS/RL_UROP-master/Cylinder2DFlowControlWithRL/saver_data/TQC27FStraineval_model_988000_steps.zip'
    vecnorm_path = '/rds/general/user/jz1720/home/TQCPM27FS/RL_UROP-master/Cylinder2DFlowControlWithRL/saver_data/TQC27FStraineval_model_vecnormalize_988000_steps.pkl'
    agent = TQC.load(saver_restore)
    env = SubprocVecEnv([resume_env(nb_actuations,i) for i in range(1)], start_method='spawn')
    env = VecFrameStack(env, n_stack=27)
    env = VecNormalize.load(venv=env, load_path=vecnorm_path)
    #model.set_env(env)
    state = env.reset()
    #example_environment.render = True
    #model.learn(15000000)
    action_step_size = simulation_duration / nb_actuations  # Duration of 1 train episode / actions in 1 episode
    single_run_duration = 700  # In non-dimensional time
    action_steps = int(single_run_duration / action_step_size)
    #evaluate_policy(model, env, n_eval_episodes=1,deterministic=True)
    #indfernals = agent.initial_internals()
    for k in range(action_steps):
        action, _ = agent.predict(state, deterministic=True)
        state, rw, done, _ = env.step(action)
