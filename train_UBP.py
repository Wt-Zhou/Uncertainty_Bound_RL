import os

import gym
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Agent.UBP import UBP
from TestScenario import CarEnv_02_Intersection, CarEnv_02_Intersection_fixed
from TestScenario_new import CarEnv_03_Ramp_Merge, CarEnv_04_Ramp_Merge

env = CarEnv_02_Intersection_fixed()
model = UBP()
model.learn(300000, env, load_model_step=250000)

