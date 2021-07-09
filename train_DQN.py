import gym
import numpy as np
import gym_routing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_02_Intersection, CarEnv_02_Intersection_fixed
from Agent.UBP import UBP,DQN


env = CarEnv_02_Intersection_fixed()
model = DQN()
model.learn(300000, env, load_model_step=0)

