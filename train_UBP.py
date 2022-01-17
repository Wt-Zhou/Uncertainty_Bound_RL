import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_02_Intersection, CarEnv_02_Intersection_fixed
from TestScenario_new import CarEnv_04_Ramp_Merge, CarEnv_03_Ramp_Merge

from Agent.UBP import UBP


env = CarEnv_03_Ramp_Merge()
model = UBP()
model.learn(300000, env, load_model_step=0)

