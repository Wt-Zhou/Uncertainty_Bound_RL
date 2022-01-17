import gym
import numpy as np

from TestScenario import CarEnv_02_Intersection, CarEnv_02_Intersection_fixed
from TestScenario_new import CarEnv_04_Ramp_Merge, CarEnv_03_Ramp_Merge

from Agent.UBP import UBP,DQN


# env = CarEnv_03_Ramp_Merge()
# model = UBP()

# # Different Training Steps
# steps_each_test = 500
# var_thres = 0.05

# model_list = [290000]
# uncertainty_thres_list = [0,1,2,3,4,5,6,7,8,9,10]
# visited_time_thres_list = [40]
# model.test(model_list=model_list, test_steps=steps_each_test, uncertainty_thres_list=uncertainty_thres_list, env=env, var_thres=var_thres, visited_time_thres_list=visited_time_thres_list)

# model_list = [200000]
# uncertainty_thres_list = [4,5]
# visited_time_thres_list = [0,20,40,60,80,100]
# model.test(model_list=model_list, test_steps=steps_each_test, uncertainty_thres_list=uncertainty_thres_list, env=env, var_thres=var_thres, visited_time_thres_list=visited_time_thres_list)

# model_list = [0,50000,100000,150000,250000,300000,200000]
# uncertainty_thres_list = [4]
# visited_time_thres_list = [40]
# model.test(model_list=model_list, test_steps=steps_each_test, uncertainty_thres_list=uncertainty_thres_list, env=env, var_thres=var_thres, visited_time_thres_list=visited_time_thres_list)

# print("Finish UBP test")
env = CarEnv_03_Ramp_Merge()
steps_each_test = 500
model_list = [0,50000,100000,150000, 200000, 240000]
model = DQN()
model.test_dqn(model_list=model_list, test_steps=steps_each_test, env=env)
print("Finish DQN test")
