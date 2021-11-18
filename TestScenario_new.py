import glob
import os
import sys
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    # sys.path.append(glob.glob("/home/icv/.local/lib/python3.6/site-packages/")[0])
except IndexError:
	pass

import carla
import time
import numpy as np
import math
import random
import gym
import cv2
import threading
from random import randint
from carla import Location, Rotation, Transform, Vector3D, VehicleControl
from collections import deque
from tqdm import tqdm
from gym import core, error, spaces, utils
from gym.utils import seeding
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

from Agent.dynamic_map import Lanepoint, Lane, Vehicle
from Agent.tools import *

MAP_NAME = 'Town04'
OBSTACLES_CONSIDERED = 3 # For t-intersection in Town02


global start_point
start_point = Transform()
start_point.location.x = 115.6
start_point.location.y = -8.75
start_point.location.z = 12
start_point.rotation.pitch = 0
start_point.rotation.yaw = -220
start_point.rotation.roll = 0


global goal_point
goal_point = Transform()
goal_point.location.x = 10
goal_point.location.y = 6
goal_point.location.z = 0
goal_point.rotation.pitch = 0
goal_point.rotation.yaw = 0 
goal_point.rotation.roll = 0

class CarEnv_04_Ramp_Merge:

    def __init__(self):
        
        # CARLA settings
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()

        if self.world.get_map().name != 'Town04':
            self.world = self.client.load_world('Town04')
        self.world.set_weather(carla.WeatherParameters(cloudiness=20, precipitation=10.0, sun_altitude_angle=50.0))
        settings = self.world.get_settings()
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = 0.2 # Warning: When change simulator, the delta_t in controller should also be change.
        settings.substepping = True
        settings.max_substep_delta_time = 0.02  # fixed_delta_seconds <= max_substep_delta_time * max_substeps
        settings.max_substeps = 10
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.free_traffic_lights(self.world)

        self.tm = self.client.get_trafficmanager(8000)
        # self.tm.set_hybrid_physics_mode(True)
        # self.tm.set_hybrid_physics_radius(50)
        self.tm.set_random_device_seed(0)

        actors = self.world.get_actors().filter('vehicle*')
        for actor in actors:
            actor.destroy()

        # Generate Reference Path
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)
        self.global_routing()

        # RL settingss
        self.action_space = spaces.Discrete(8) # len(fplist + 2) 0 for rule, 1 for brake 
        self.low  = np.array([133,  198, -2, -2,125, 189, -1, -1, 128, 195, -2, -1, 125, 195, -2,-1], dtype=np.float64)
        self.high = np.array([137,  203, 1, 1, 130, 194,  2,  1,  132,  200,  1,  2,  130,  200 , 1, 2], dtype=np.float64)    
        # self.low  = np.array([130,  190, -6, -6,120, 189, -1, -1, 120, 185, -5, -1, 120, 185, -5,-1], dtype=np.float64)
        # self.high = np.array([150,  220, 1, 1, 140, 194,  5,  1,  140,  205,  1,  5,  140,  205 , 1, 5], dtype=np.float64)    
        # low  = np.array([100,  170, -15, -15,50, 189, -15, -15, 50, 185, -15, -15, 110, 185, -15,-15])
        # high = np.array([150,  220, 15, 15, 155, 194,  15,  15,  150,  205,  15,  15,  190,  205 , 15, 15])    
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # Spawn Ego Vehicle
        global start_point
        self.ego_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.mercedes-benz.coupe'))
        if self.ego_vehicle_bp.has_attribute('color'):
            color = '255,0,0'
            self.ego_vehicle_bp.set_attribute('color', color)
            self.ego_vehicle_bp.set_attribute('role_name', "ego_vehicle")
        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, start_point)
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.ego_collision_sensor = self.world.spawn_actor(collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))
        self.ego_vehicle_collision_sign = False
        self.stuck_time = 0
        
        self.env_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle.mercedes-benz.coupe'))
        if self.env_vehicle_bp.has_attribute('color'):
            color = '0,0,255'
            self.env_vehicle_bp.set_attribute('color', color)
        if self.env_vehicle_bp.has_attribute('driver_id'):
            driver_id = random.choice(self.env_vehicle_bp.get_attribute('driver_id').recommended_values)
            self.env_vehicle_bp.set_attribute('driver_id', driver_id)
        self.env_vehicle_bp.set_attribute('role_name', 'autopilot')

        # Control Env Vehicle
        self.has_set = np.zeros(1000000)
        self.stopped_time = np.zeros(1000000)   

        # Record
        self.log_dir = "record.txt"
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

        # Case
        self.init_case()
        self.case_id = 0
       
    def free_traffic_lights(self, carla_world):
        traffic_lights = carla_world.get_actors().filter('*traffic_light*')
        for tl in traffic_lights:
            tl.set_green_time(5)
            tl.set_red_time(5)

    def global_routing(self):
        global goal_point
        global start_point

        start = start_point
        goal = goal_point
        print("Calculating route to x={}, y={}, z={}".format(
                goal.location.x,
                goal.location.y,
                goal.location.z))
        
        dao = GlobalRoutePlannerDAO(self.world.get_map(), 1)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        current_route = grp.trace_route(carla.Location(start.location.x,
                                                start.location.y,
                                                start.location.z),
                                carla.Location(goal.location.x,
                                                goal.location.y,
                                                goal.location.z))
        t_array = []
        self.ref_path = Lane()
        for wp in current_route:
            lanepoint = Lanepoint()
            lanepoint.position.x = wp[0].transform.location.x + 2.0
            lanepoint.position.y = wp[0].transform.location.y
            self.ref_path.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path.central_path_array = np.array(t_array)
        self.ref_path.speed_limit = 60/3.6 # m/s

        ref_path_ori = convert_path_to_ndarray(self.ref_path.central_path)
        self.ref_path_array = dense_polyline2d(ref_path_ori, 2)
        self.ref_path_tangets = np.zeros(len(self.ref_path_array))

    def ego_vehicle_stuck(self, stay_thres = 5):        
        ego_vehicle_velocity = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)
        if ego_vehicle_velocity < 0.1:
            pass
        else:
            self.stuck_time = time.time()

        if time.time() - self.stuck_time > stay_thres:
            return True
        return False

    def ego_vehicle_pass(self):
        global goal_point
        ego_location = self.ego_vehicle.get_location()
        if ego_location.distance(goal_point.location) < 35:
            return True
        else:
            return False

    def ego_vehicle_collision(self, event):
        self.ego_vehicle_collision_sign = True

    def wrap_state(self):
        # state = [0 for i in range((OBSTACLES_CONSIDERED + 1) * 4)]
        state  = np.array([100,  220, -15, -15,115, 192, 0, 0, 150, 183, 0, 0, 150, 183, 0,0], dtype=np.float64)

        ego_vehicle_state = Vehicle()
        ego_vehicle_state.x = self.ego_vehicle.get_location().x
        ego_vehicle_state.y = self.ego_vehicle.get_location().y
        ego_vehicle_state.v = math.sqrt(self.ego_vehicle.get_velocity().x ** 2 + self.ego_vehicle.get_velocity().y ** 2 + self.ego_vehicle.get_velocity().z ** 2)

        ego_vehicle_state.yaw = self.ego_vehicle.get_transform().rotation.yaw / 180.0 * math.pi # Transfer to rad
        ego_vehicle_state.yawdt = self.ego_vehicle.get_angular_velocity()

        ego_vehicle_state.vx = ego_vehicle_state.v * math.cos(ego_vehicle_state.yaw)
        ego_vehicle_state.vy = ego_vehicle_state.v * math.sin(ego_vehicle_state.yaw)

        # Ego state
        state[0] = ego_vehicle_state.x 
        state[1] = ego_vehicle_state.y 
        state[2] = ego_vehicle_state.vx 
        state[3] = ego_vehicle_state.vy 

        # Obs state
        closest_obs = []
        closest_obs = self.found_closest_obstacles_t_intersection()
        i = 0
        for obs in closest_obs: 
            if i < OBSTACLES_CONSIDERED:
                if obs[0] != 0:
                    state[(i+1)*4+0] = obs[0] 
                    state[(i+1)*4+1] = obs[1] 
                    state[(i+1)*4+2] = obs[2]
                    state[(i+1)*4+3] = obs[3]
                i = i+1
            else:
                break
        
        return state

    def found_closest_obstacles_ramp(self):
        obs_tuples = []
        for obs in self.world.get_actors().filter('vehicle*'): 
            # Calculate distance
            p1 = np.array([self.ego_vehicle.get_location().x ,  self.ego_vehicle.get_location().y])
            p2 = np.array([obs.get_location().x , obs.get_location().y])
            p3 = p2 - p1
            p4 = math.hypot(p3[0],p3[1])
            
            # Obstacles too far
            one_obs = (obs.get_location().x, obs.get_location().y, obs.get_velocity().x, obs.get_velocity().y, p4, obs.get_transform().rotation.yaw)
            if p4 > 0:
                obs_tuples.append(one_obs)
        
        closest_obs = []
        fake_obs = [0 for i in range(11)]  #len(one_obs)
        for i in range(0, OBSTACLES_CONSIDERED ,1): # 3 obs
            closest_obs.append(fake_obs)
        
        # Sort by distance
        sorted_obs = sorted(obs_tuples, key=lambda obs: obs[4])   

        put_1st = False
        put_2nd = False
        put_3rd = False
        for obs in sorted_obs:
            if obs[0] > 115 and obs[0] < 155 and obs[1] < 195 and obs[1] > 190 and obs[2] > 0 and math.fabs(obs[3]) < 0.5 and put_1st == False and (obs[5] < 60 and obs[5] > -60):
                closest_obs[0] = obs 
                put_1st = True
                continue
            if obs[1] > 185 and obs[1] < 205 and obs[2] < 0.5 and (obs[1] < self.ego_vehicle.get_location().y - 3 or obs[1] < 194) \
                    and obs[0] <  self.ego_vehicle.get_location().x - 5 and obs[0] > 110 and put_2nd == False and (obs[5] > 60 or obs[5] < -60):
                closest_obs[1] = obs
                put_2nd = True
                continue
            if obs[1] > 185 and obs[1] < 205 and obs[2] < 0.5 and (obs[1] < self.ego_vehicle.get_location().y - 3 or obs[1] < 194) \
                     and obs[0] >  self.ego_vehicle.get_location().x -5 and obs[0] < 150 and put_3rd == False and (obs[5] > 60 or obs[5] < -60):
                closest_obs[2] = obs
                put_3rd = True
                continue
            else:
                continue
        return closest_obs
                                            
    def record_information_txt(self):
        if self.task_num > 0:
            stuck_rate = float(self.stuck_num) / float(self.task_num)
            collision_rate = float(self.collision_num) / float(self.task_num)
            pass_rate = 1 - ((float(self.collision_num) + float(self.stuck_num)) / float(self.task_num))
            fw = open(self.log_dir, 'a')   
            # Write num
            fw.write(str(self.task_num)) 
            fw.write(", ")
            fw.write(str(self.case_id)) 
            fw.write(", ")
            fw.write(str(self.stuck_num)) 
            fw.write(", ")
            fw.write(str(self.collision_num)) 
            fw.write(", ")
            fw.write(str(stuck_rate)) 
            fw.write(", ")
            fw.write(str(collision_rate)) 
            fw.write(", ")
            fw.write(str(pass_rate)) 
            fw.write("\n")
            fw.close()               
            print("[CARLA]: Record To Txt: All", self.task_num, self.stuck_num, self.collision_num, self.case_id )

    def clean_task_nums(self):
        self.task_num = 0
        self.stuck_num = 0
        self.collision_num = 0

    def reset(self):    
        # Control Env Elements
        self.spawn_fixed_veh()

        # Ego vehicle
        if self.ego_vehicle is not None:
            self.ego_collision_sensor.destroy()
            self.ego_vehicle.destroy()

        global start_point
        self.ego_vehicle = self.world.spawn_actor(self.ego_vehicle_bp, start_point)

        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.ego_collision_sensor = self.world.spawn_actor(collision_bp, Transform(), self.ego_vehicle, carla.AttachmentType.Rigid)
        self.ego_collision_sensor.listen(lambda event: self.ego_vehicle_collision(event))
        self.ego_vehicle_collision_sign = False

        self.world.tick() 

        # State
        state = self.wrap_state()

        # Record
        self.record_information_txt()
        self.task_num += 1
        self.case_id += 1

        return state

    def step(self, action):
        # Control ego vehicle
        print("action[0]",action[0])
        throttle = max(0,float(action[0]))  # range [0,1]
        brake = max(0,-float(action[0])) # range [0,1]
        steer = action[1] # range [-1,1]
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = throttle, brake = brake, steer = steer))
        self.world.tick()

        # State
        state = self.wrap_state()

        # Step reward
        reward = 0
        # If finish
        done = False
        if self.ego_vehicle_collision_sign:
            self.collision_num += + 1
            done = True
            reward = 0
            print("[CARLA]: Collision!")
        
        if self.ego_vehicle_pass():
            done = True
            reward = 1
            print("[CARLA]: Successful!")

        elif self.ego_vehicle_stuck():
            self.stuck_num += 1
            reward = -0.0
            done = True
            print("[CARLA]: Stuck!")


        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        for vehicle in vehicle_list:  
            self.tm.ignore_signs_percentage(vehicle, 100)
            self.tm.ignore_lights_percentage(vehicle, 100)
            self.tm.ignore_walkers_percentage(vehicle, 0)
            self.tm.set_percentage_keep_right_rule(vehicle,100) # it can make the actor go forward, dont know why
            self.tm.auto_lane_change(vehicle, True)
            self.tm.distance_to_leading_vehicle(vehicle, 10)

        return state, reward, done, None

    def init_case(self):
        self.case_list = []

        # one vehicle front
        for i in range(0,10):
            spawn_vehicles = []
            transform = Transform()
            transform.location.x = 92 - i * 5
            transform.location.y = 6
            transform.location.z = 12
            transform.rotation.pitch = 0
            transform.rotation.yaw = -180
            transform.rotation.roll = 0
            spawn_vehicles.append(transform)
            self.case_list.append(spawn_vehicles)

        # one vehicle behind
        for i in range(0,10):
            spawn_vehicles = []
            transform = Transform()
            transform.location.x = 88 + i * 5 
            transform.location.y = 6
            transform.location.z = 12
            transform.rotation.pitch = 0
            transform.rotation.yaw = -180
            transform.rotation.roll = 0
            spawn_vehicles.append(transform)
            self.case_list.append(spawn_vehicles)

        # two vehicles
        for i in range(0,10):
            for j in range(0,10):
                spawn_vehicles = []
                transform = Transform()
                transform.location.x = 92 - i * 5 
                transform.location.y = 6
                transform.location.z = 12
                transform.rotation.pitch = 0
                transform.rotation.yaw = -180
                transform.rotation.roll = 0
                spawn_vehicles.append(transform)
                transform = Transform()
                transform.location.x = 88 + j * 5
                transform.location.y = 6
                transform.location.z = 12
                transform.rotation.pitch = 0
                transform.rotation.yaw = -180
                transform.rotation.roll = 0
                spawn_vehicles.append(transform)
                self.case_list.append(spawn_vehicles)

        # 3 vehicles
        for i in range(0,10):
            for j in range(0,10):
                for k in range (0,10):
                    spawn_vehicles = []
                    transform.location.x = 92 - i * 5 
                    transform.location.y = 6
                    transform.location.z = 12
                    transform.rotation.pitch = 0
                    transform.rotation.yaw = -180
                    transform.rotation.roll = 0
                    spawn_vehicles.append(transform)
                    
                    transform = Transform()
                    transform.location.x = 88 + j * 5
                    transform.location.y = 6
                    transform.location.z = 12
                    transform.rotation.pitch = 0
                    transform.rotation.yaw = -180
                    transform.rotation.roll = 0
                    spawn_vehicles.append(transform)
                    
                    transform = Transform()
                    transform.location.x = 50 + j * 5
                    transform.location.y = 10
                    transform.location.z = 12
                    transform.rotation.pitch = 0
                    transform.rotation.yaw = -180
                    transform.rotation.roll = 0
                    spawn_vehicles.append(transform)
                    self.case_list.append(spawn_vehicles)

        print("How many Cases?",len(self.case_list))

    def spawn_fixed_veh(self):
        if self.case_id >= len(self.case_list):
            self.case_id = 1
        actor_list = self.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor
        synchronous_master = True

        for vehicle in vehicle_list:
            if vehicle.attributes['role_name'] != "ego_vehicle" :
                vehicle.destroy()

        batch = []
        print("Case_id",self.case_id)

        for transform in self.case_list[self.case_id - 1]:
            batch.append(SpawnActor(self.env_vehicle_bp, transform).then(SetAutopilot(FutureActor, True)))
    
        self.client.apply_batch_sync(batch, synchronous_master)

        




