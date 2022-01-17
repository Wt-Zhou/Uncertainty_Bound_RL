import argparse

import numpy as np
import math
import os
import os.path as osp
import tensorflow as tf
import tempfile
import time
import random
import _thread
import baselines.common.tf_util as U
import random

from tqdm import tqdm
from rtree import index as rindex
from collections import deque
from scipy import stats
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from Agent.model import dqn_model, bootstrap_model
from Agent.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.controller import Controller
from Agent.dynamic_map import DynamicMap
from Agent.actions import LaneAction

class DQN(object):

    def __init__(self):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 


    def parse_args(self):
        parser = argparse.ArgumentParser("DQN experiments for Atari games")
        # Environment
        parser.add_argument("--env", type=str, default="DQN", help="name of the game")
        parser.add_argument("--seed", type=int, default=42, help="which seed to use")
        parser.add_argument("--decision_count", type=int, default=5, help="how many steps for a decision")
        # Core DQN parameters
        parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
        parser.add_argument("--train-buffer-size", type=int, default=int(1e8), help="train buffer size")
        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for Adam optimizer")
        parser.add_argument("--num-steps", type=int, default=int(4e7), help="total number of steps to run the environment for")
        parser.add_argument("--batch-size", type=int, default=64, help="number of transitions to optimize at the same time")
        parser.add_argument("--learning-freq", type=int, default=20, help="number of iterations between every optimization step")
        parser.add_argument("--target-update-freq", type=int, default=50, help="number of iterations between every target network update") #10000
        parser.add_argument("--learning-starts", type=int, default=50, help="when to start learning") 
        parser.add_argument("--gamma", type=float, default=0.995, help="the gamma of q update") 
        parser.add_argument("--bootstrapped-data-sharing-probability", type=float, default=0.8, help="bootstrapped_data_sharing_probability") 
        parser.add_argument("--bootstrapped-heads-num", type=int, default=10, help="bootstrapped head num of networks") 
        parser.add_argument("--learning-repeat", type=int, default=10, help="learn how many times from one sample of RP") 
        # Bells and whistles
        boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
        boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
        boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
        boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
        parser.add_argument("--prioritized-alpha", type=float, default=0.9, help="alpha parameter for prioritized replay buffer")
        parser.add_argument("--prioritized-beta0", type=float, default=0.1, help="initial value of beta parameters for prioritized replay")
        parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
        # Checkpointing
        parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
        parser.add_argument("--save-azure-container", type=str, default=None,
                            help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
        parser.add_argument("--save-freq", type=int, default=10000, help="save model once every time this many iterations are completed")
        boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
        return parser.parse_args()

    def maybe_save_model(self, savedir, state):
        """This function checkpoints the model and state of the training algorithm."""
        if savedir is None:
            return
        start_time = time.time()
        model_dir = "model-{}".format(state["num_iters"])
        U.save_state(os.path.join(savedir, model_dir, "saved"))
        state_dir = "training_state.pkl-{}".format(state["num_iters"]) + ".zip"
        relatively_safe_pickle_dump(state, os.path.join(savedir, state_dir), compression=True)
        logger.log("Saved model in {} seconds\n".format(time.time() - start_time))

    def maybe_load_model(self, savedir, model_step):
        """Load model if present at the specified path."""
        if savedir is None:
            return
        model_dir = "training_state.pkl-{}".format(model_step) + ".zip"
        # state_path = os.path.join(os.path.join(savedir, 'training_state.pkl-100028.zip'))
        state_path = os.path.join(os.path.join(savedir, model_dir))
        found_model = os.path.exists(state_path)
        if found_model:
            state = pickle_load(state_path, compression=True)
            model_dir = "model-{}".format(state["num_iters"])
            U.load_state(os.path.join(savedir, model_dir, "saved"))
            logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
            return state

    def test_dqn(self, model_list, test_steps, env):
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True

        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
            # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters
            
            learning_rate = args.lr # Maybe Segmented

            U.initialize()
            num_iters = 0

            for model_step in model_list:
                # Load the model
                state = self.maybe_load_model(savedir, model_step)
                if state is not None:
                    num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
        
                start_time, start_steps = None, None
                test_iters = 0

                obs = env.reset()
                self.trajectory_planner.clear_buff(clean_csp=False)

                # Test
                while test_iters < test_steps:
                    num_iters += 1
                    obs = np.array(obs)
                    # Rule-based Planner
                    self.dynamic_map.update_map_from_obs(obs, env)
                    rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                    # DQN Action
                    q_list = q_values_dqn(obs[None])
                    action = np.array(np.where(q_list[0]==np.max(q_list[0]))[0])

                    print("[Bootstrap DQN]: Obs",obs.tolist())
                    print("[Bootstrap DQN]: DQN Action",action)
                    print("[Bootstrap DQN]: DQN value",q_list[0])

                    # Control
                    trajectory = self.trajectory_planner.trajectory_update_UBP(action[0], rule_trajectory)
                    for i in range(args.decision_count):
                        control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                        output_action = [control_action.acc, control_action.steering]
                        new_obs, rew, done, info = env.step(output_action)
                        if done:
                            break
                        self.dynamic_map.update_map_from_obs(new_obs, env)

                    obs = new_obs
                    if done:
                        obs = env.reset()
                        self.trajectory_planner.clear_buff(clean_csp=False)
                        test_iters += 1

                    # Record Data    
                self.record_test_data(model_step, 2333, 2333, env)

    def learn(self, total_timesteps, env, load_model_step):      
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True


        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
        # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters

            learning_rate = args.lr # maybe Segmented

            U.initialize()
            update_target_dqn()

            # Load the model
            state = self.maybe_load_model(savedir, load_model_step)
            if state is not None:
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
                load_model_step=load_model_step
            else:
                load_model_step = 0
            
            start_time, start_steps = None, None

            obs = env.reset()
            self.trajectory_planner.clear_buff(clean_csp=False)
            decision_count = 0
            
            for num_iters in tqdm(range(load_model_step, total_timesteps + 1), unit='steps'):
                obs = np.array(obs)

                # Rule-based Planner
                self.dynamic_map.update_map_from_obs(obs, env)
                rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                # Bootstapped Action
                dqn_q = q_values_dqn(obs[None])
                optimal_action = np.array(np.where(dqn_q[0]==np.max(dqn_q[0]))[0][0])
                random_action = random.randint(0,7)

                if random.uniform(0,1) < 0.2: # epsilon-greddy
                    action = random_action
                else:
                    action = optimal_action

                print("[DQN]: Obs",obs.tolist())
                print("[DQN]: Action",action, random_action)
                # print("[DQN]: DQN Q-value",dqn_q)

                # Control
                trajectory = self.trajectory_planner.trajectory_update_UBP(action, rule_trajectory)
                for i in range(args.decision_count):
                    control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                    output_action = [control_action.acc, control_action.steering]
                    new_obs, rew, done, info = env.step(output_action)
                    if done:
                        break
                    self.dynamic_map.update_map_from_obs(new_obs, env)
                    
                mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
                replay_buffer.add(obs, action, rew, new_obs, float(done), mask)
                obs = new_obs
                if done:
                    obs = env.reset()
                    self.trajectory_planner.clear_buff(clean_csp=False)

                if (num_iters > args.learning_starts and
                        num_iters % args.learning_freq == 0):
                    # Sample a bunch of transitions from replay buffer
                    if args.prioritized:
                        # Update rl
                        if replay_buffer.__len__() > args.batch_size:
                            for i in range(args.learning_repeat):
                                print("[DQN]: Learning")
                                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters), count_train=True)
                                (obses_t, actions, rewards, obses_tp1, dones, masks, train_time, weights, batch_idxes) = experience
                                td_errors_dqn, q_t_selected_dqn, q_t_selected_target_dqn, qt_dqn = train_dqn(obses_t, actions, rewards, obses_tp1, dones, masks, weights, learning_rate)
                                # Update the priorities in the replay buffer
                                new_priorities = np.abs(td_errors_dqn) + args.prioritized_eps

                                replay_buffer.update_priorities(batch_idxes, new_priorities)
                    
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                        weights = np.ones_like(rewards)
                    
                    
                # Update target network.
                if num_iters % args.target_update_freq == 0:
                    print("[DQN]: Update target network")
                    update_target_dqn()

                start_time, start_steps = time.time(), 0

                # Save the model and training state.
                if num_iters >= 0 and num_iters % args.save_freq == 0:
                    print("[DQN]: Save model")
                    self.maybe_save_model(savedir, {
                        'replay_buffer': replay_buffer,
                        'num_iters': num_iters,
                    })
                    save_model = False
                    
            print("[DQN]: Finish Training, Save model")
            self.maybe_save_model(savedir, {
                'replay_buffer': replay_buffer,
                'num_iters': num_iters,
            })

    def record_test_data(self, model_step, uncertainty_thres, visited_time_thres, env):
        with open("Test_data-{}".format(model_step) + "-{}".format(uncertainty_thres) + "-{}".format(visited_time_thres) + ".txt", 'a') as fw:
            fw.write(str(env.task_num - 1))  # The num will be add 1 in reset()
            fw.write(", ")
            fw.write(str(env.stuck_num)) 
            fw.write(", ")
            fw.write(str(env.collision_num)) 
            fw.write("\n")
            fw.close()    
        env.clean_task_nums()


class UBP(object):

    def __init__(self):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        self.bound = Bound()
        self.rtree = RTree()

    def parse_args(self):
        parser = argparse.ArgumentParser("DQN experiments for Atari games")
        # Environment
        parser.add_argument("--env", type=str, default="Seaquest", help="name of the game")
        parser.add_argument("--seed", type=int, default=42, help="which seed to use")
        parser.add_argument("--decision_count", type=int, default=5, help="how many steps for a decision")
        # Core DQN parameters
        parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
        parser.add_argument("--train-buffer-size", type=int, default=int(1e8), help="train buffer size")
        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for Adam optimizer")
        parser.add_argument("--num-steps", type=int, default=int(4e7), help="total number of steps to run the environment for")
        parser.add_argument("--batch-size", type=int, default=64, help="number of transitions to optimize at the same time")
        parser.add_argument("--learning-freq", type=int, default=20, help="number of iterations between every optimization step")
        parser.add_argument("--target-update-freq", type=int, default=50, help="number of iterations between every target network update") #10000
        parser.add_argument("--learning-starts", type=int, default=50, help="when to start learning") 
        parser.add_argument("--gamma", type=float, default=0.995, help="the gamma of q update") 
        parser.add_argument("--bootstrapped-data-sharing-probability", type=float, default=0.8, help="bootstrapped_data_sharing_probability") 
        parser.add_argument("--bootstrapped-heads-num", type=int, default=10, help="bootstrapped head num of networks") 
        parser.add_argument("--learning-repeat", type=int, default=10, help="learn how many times from one sample of RP") 
        # Bells and whistles
        boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
        boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
        boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
        boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
        parser.add_argument("--prioritized-alpha", type=float, default=0.9, help="alpha parameter for prioritized replay buffer")
        parser.add_argument("--prioritized-beta0", type=float, default=0.1, help="initial value of beta parameters for prioritized replay")
        parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
        # Checkpointing
        parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
        parser.add_argument("--save-azure-container", type=str, default=None,
                            help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
        parser.add_argument("--save-freq", type=int, default=10000, help="save model once every time this many iterations are completed")
        boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
        return parser.parse_args()

    def maybe_save_model(self, savedir, state):
        """This function checkpoints the model and state of the training algorithm."""
        if savedir is None:
            return
        start_time = time.time()
        model_dir = "model-{}".format(state["num_iters"])
        U.save_state(os.path.join(savedir, model_dir, "saved"))
        state_dir = "training_state.pkl-{}".format(state["num_iters"]) + ".zip"
        relatively_safe_pickle_dump(state, os.path.join(savedir, state_dir), compression=True)
        logger.log("Saved model in {} seconds\n".format(time.time() - start_time))

    def maybe_load_model(self, savedir, model_step):
        """Load model if present at the specified path."""
        if savedir is None:
            return
        model_dir = "training_state.pkl-{}".format(model_step) + ".zip"
        # state_path = os.path.join(os.path.join(savedir, 'training_state.pkl-100028.zip'))
        state_path = os.path.join(os.path.join(savedir, model_dir))
        found_model = os.path.exists(state_path)
        if found_model:
            state = pickle_load(state_path, compression=True)
            model_dir = "model-{}".format(state["num_iters"])
            U.load_state(os.path.join(savedir, model_dir, "saved"))
            logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
            return state

    def test(self, model_list, test_steps, uncertainty_thres_list, env, var_thres, visited_time_thres_list):
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True

        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
            # Create training graph and replay buffer
            act_ubp, train, update_target, q_values = deepq.build_train(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                q_func=bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            # act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
            #     make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
            #     original_dqn=dqn_model,
            #     num_actions=env.action_space.n,
            #     optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            #     gamma=args.gamma,
            #     grad_norm_clipping=10,
            #     double_q=args.double_q
            # )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters
            
            learning_rate = args.lr # Maybe Segmented

            U.initialize()
            update_target()
            num_iters = 0

            for model_step in model_list:
                # Load the model
                state = self.maybe_load_model(savedir, model_step)
                if state is not None:
                    num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
        
                # Using the latest train buffer to update rtree FIXME
                self.rtree.__init__()
                self.rtree.update_with_replay_buffer(replay_buffer)
                for uncertainty_thres in uncertainty_thres_list:
                    for visited_time_thres in visited_time_thres_list:
                        start_time, start_steps = None, None
                        test_iters = 0

                        obs = env.reset()
                        self.trajectory_planner.clear_buff(clean_csp=False)

                        # Test
                        while test_iters < test_steps:
                            num_iters += 1
                            obs = np.array(obs)
                            # Rule-based Planner
                            self.dynamic_map.update_map_from_obs(obs, env)
                            rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                            # Bootstapped Action
                            q_list = q_values(obs[None])
                            action = self.bound.act_test_bootstrap_optimal(obs, q_values, uncertainty_thres, var_thres, self.rtree, visited_time_thres) 
                            print("[Bootstrap DQN]: Obs",obs.tolist())
                            print("[Bootstrap DQN]: Action",action)

                            # Control
                            trajectory = self.trajectory_planner.trajectory_update_UBP(action, rule_trajectory)
                            for i in range(args.decision_count):
                                control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                                output_action = [control_action.acc, control_action.steering]
                                new_obs, rew, done, info = env.step(output_action)
                                if done:
                                    break
                                self.dynamic_map.update_map_from_obs(new_obs, env)

                            obs = new_obs
                            if done:
                                self.record_termianl_data(model_step, obs, action, rew, q_values, self.rtree) # before update obs
                                obs = env.reset()
                                self.trajectory_planner.clear_buff(clean_csp=False)
                                test_iters += 1

                            # Record Data    
                        self.record_test_data(model_step, uncertainty_thres, visited_time_thres, env)
    
    def test_dqn(self, model_list, test_steps, env):
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True

        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
            # Create training graph and replay buffer
            act_ubp, train, update_target, q_values = deepq.build_train(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                q_func=bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters
            
            learning_rate = args.lr # Maybe Segmented

            U.initialize()
            update_target()
            num_iters = 0

            for model_step in model_list:
                # Load the model
                state = self.maybe_load_model(savedir, model_step)
                if state is not None:
                    num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
        
                start_time, start_steps = None, None
                test_iters = 0

                obs = env.reset()
                self.trajectory_planner.clear_buff(clean_csp=False)

                # Test
                while test_iters < test_steps:
                    num_iters += 1
                    obs = np.array(obs)
                    # Rule-based Planner
                    self.dynamic_map.update_map_from_obs(obs, env)
                    rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                    # DQN Action
                    q_list = q_values_dqn(obs[None])
                    action = np.array(np.where(q_list[0]==np.max(q_list[0]))[0][0])

                    print("[Bootstrap DQN]: Obs",obs.tolist())
                    print("[Bootstrap DQN]: DQN Action",action)
                    print("[Bootstrap DQN]: DQN value",q_list[0])

                    # Control
                    trajectory = self.trajectory_planner.trajectory_update_UBP(action, rule_trajectory)
                    for i in range(args.decision_count):
                        control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                        output_action = [control_action.acc, control_action.steering]
                        new_obs, rew, done, info = env.step(output_action)
                        if done:
                            break
                        self.dynamic_map.update_map_from_obs(new_obs, env)

                    obs = new_obs
                    if done:
                        obs = env.reset()
                        self.trajectory_planner.clear_buff(clean_csp=False)
                        test_iters += 1

                    # Record Data    
                self.record_test_data(model_step, 2333, 2333, env)

    def learn(self, total_timesteps, env, load_model_step):      
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True


        if args.seed > 0:
            set_global_seeds(args.seed)

        with U.make_session(120) as sess:
        # Create training graph and replay buffer
            act_ubp, train, update_target, q_values = deepq.build_train(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                q_func=bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters

            learning_rate = args.lr # maybe Segmented

            U.initialize()
            update_target()
            num_iters = 0

            # Load the model
            state = self.maybe_load_model(savedir, load_model_step)
            if state is not None:
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
                load_model_step = load_model_step
            else:
                load_model_step = 0
            
            self.rtree.update_with_replay_buffer(replay_buffer)

            start_time, start_steps = None, None

            obs = env.reset()
            self.trajectory_planner.clear_buff(clean_csp=False)

            # Main training loop
            random_head = np.random.randint(args.bootstrapped_heads_num)        #Initial head initialisation
            
            decision_count = 0

            for num_iters in tqdm(range(load_model_step, total_timesteps + 1), unit='steps'):

                obs = np.array(obs)

                # Rule-based Planner
                self.dynamic_map.update_map_from_obs(obs, env)
                rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                # Bootstapped Action
                q_list = q_values(obs[None])
                bootstrap_action = np.where(q_list[random_head][0]==np.max(q_list[random_head][0]))   

                random_action = random.randint(0,8)
                if random.uniform(0,1) < 0.5 * (300000 - num_iters) / 300000: # epsilon-greddy
                    bootstrap_action = random_action
      
                action = self.bound.act_train(obs, q_values, bootstrap_action, self.rtree, num_iters) 
            

                print("[Bootstrap DQN]: Obs",obs.tolist())
                # print("[Bootstrap DQN]: Action",action)
                # print("[Bootstrap DQN]: Visited Times",self.rtree.calculate_visited_times(obs,0))

                # Control
                trajectory = self.trajectory_planner.trajectory_update_UBP(action, rule_trajectory)
                for i in range(args.decision_count):
                    control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                    output_action = [control_action.acc, control_action.steering]
                    new_obs, rew, done, info = env.step(output_action)
                    if done:
                        break
                    self.dynamic_map.update_map_from_obs(new_obs, env)
                    
                mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
                replay_buffer.add(obs, action, rew, new_obs, float(done), mask)
                obs = new_obs
                if done:
                    random_head = np.random.randint(args.bootstrapped_heads_num)
                    obs = env.reset()
                    self.trajectory_planner.clear_buff(clean_csp=False)

                if (num_iters > args.learning_starts and
                        num_iters % args.learning_freq == 0):
                    # Sample a bunch of transitions from replay buffer
                    if args.prioritized:
                        # Update rl
                        if replay_buffer.__len__() > args.batch_size:
                            for i in range(args.learning_repeat):
                                print("[Bootstrap DQN]: Learning")
                                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters), count_train=True)
                                (obses_t, actions, rewards, obses_tp1, dones, masks, train_time, weights, batch_idxes) = experience
                                self.rtree.add_data_to_rtree(experience) #FIXME
                                # Minimize the error in Bellman's equation and compute TD-error
                                # td_errors, q_t_selected, q_t_selected_target, obs_after = train(obses_t, actions, rewards, obses_tp1, dones, masks, np.ones_like(rewards), learning_rate)
                                td_errors, q_t_selected, q_t_selected_target, obs_after = train(obses_t, actions, rewards, obses_tp1, dones, masks, weights, learning_rate)
                                # Update the priorities in the replay buffer
                                new_priorities = np.abs(np.mean(td_errors, axis=0)) + args.prioritized_eps
                                replay_buffer.update_priorities(batch_idxes, new_priorities)
                    
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                        weights = np.ones_like(rewards)
                    
                    
                # Update target network.
                if num_iters % args.target_update_freq == 0:
                    print("[Bootstrap DQN]: Update target network")
                    update_target()


               
                start_time, start_steps = time.time(), 0

                # Save the model and training state.
                if num_iters >= 0 and num_iters % args.save_freq == 0:                                       
                    print("[Bootstrap DQN]: Save model")
                    self.maybe_save_model(savedir, {
                        'replay_buffer': replay_buffer,
                        'num_iters': num_iters,
                    })

    def record_test_data(self, model_step, uncertainty_thres, visited_time_thres, env):
        with open("Test_data-{}".format(model_step) + "-{}".format(uncertainty_thres) + "-{}".format(visited_time_thres) + ".txt", 'a') as fw:
            fw.write(str(self.bound.rule_times)) 
            fw.write(", ")
            fw.write(str(self.bound.rl_times)) 
            fw.write(", ")
            fw.write(str(env.task_num - 1))  # The num will be add 1 in reset()
            fw.write(", ")
            fw.write(str(env.stuck_num)) 
            fw.write(", ")
            fw.write(str(env.collision_num)) 
            fw.write("\n")
            fw.close()    
        self.bound.clean_running_times()
        env.clean_task_nums()

    def record_termianl_data(self, model_step, obs, action, rew, q_values, rtree):
        q_action_list = []
        q_list = q_values(obs[None])
        for i in range(10):
            q_action_list.append(q_list[i][0][action])

        with open("Termial_data.txt", 'a') as fw:
            fw.write(str(model_step)) 
            fw.write(", ")
            fw.write(str(obs.tolist())) 
            fw.write(", ")
            fw.write(str(action)) 
            fw.write(", ")
            fw.write(str(rew))
            fw.write(", ")
            fw.write(str(q_action_list))
            fw.write(", ")
            fw.write(str(rtree.calculate_visited_times(obs, action)))
            fw.write("\n")
            fw.close()    

    def data_uncertainty(self, env, model_list, obs_list):
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True

        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
            # Create training graph and replay buffer
            act_ubp, train, update_target, q_values = deepq.build_train(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                q_func=bootstrap_model,
                bootstrap=args.bootstrap,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            # act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
            #     make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
            #     original_dqn=dqn_model,
            #     num_actions=env.action_space.n,
            #     optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
            #     gamma=args.gamma,
            #     grad_norm_clipping=10,
            #     double_q=args.double_q
            # )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters
            
            learning_rate = args.lr # Maybe Segmented

            U.initialize()
            update_target()
            num_iters = 0
            for model in model_list:
                # Load the model
                state = self.maybe_load_model(savedir, model)
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]

                self.rtree = RTree()
                self.rtree.update_with_replay_buffer(replay_buffer)
                for obs in obs_list:
                    obs = np.array(obs)
                    action = 0

                    q_list = q_values(obs[None])
                    q_action_list = []
                    for i in range(10):
                        q_action_list.append(q_list[i][0][action])

                    mean_rule = np.mean(np.array(q_action_list))
                    var_rule = np.var(np.array(q_action_list))

                    # Calculate visited times of s,a
                    visited_times = self.rtree.calculate_visited_times(obs, action)

                    # Write q list to txt
                    print("Writing: Obs", obs, action)
                    print("Writing: Model", model)
                    print("Writing: Visited_times", visited_times)
                    print("Writing: Q_action_list", q_action_list)
                    # print("Writing: Q_value_DQN", q_values_dqn(obs[None])[0][action])
                    print("Writing: Mean and Var:",mean_rule, var_rule)
                    fw = open("data_uncertainty.txt", 'a')   
                    fw.write(str(obs.tolist()[0])) 
                    fw.write(", ")
                    fw.write(str(obs.tolist()[1])) 
                    fw.write(", ")
                    fw.write(str(model)) 
                    fw.write(", ")
                    fw.write(str(visited_times)) 
                    fw.write(", ")
                    fw.write(str(q_action_list)) 
                    fw.write(", ")
                    fw.write(str(var_rule)) 
                    fw.write("\n")
                    fw.close()       

class RTree(object):

    def __init__(self, new_count=True):
        if new_count == True:
            if osp.exists("visited_state_value.txt"):
                os.remove("visited_state_value.txt")
            if osp.exists("state_index.dat"):
                os.remove("state_index.dat")
                os.remove("state_index.idx")

            # _setup_data_saving 
            self.visited_state_counter = 0
        else:
            self.visited_state_value = np.loadtxt("visited_value.txt")
            self.visited_state_value = self.visited_state_value.tolist()
            self.visited_state_counter = len(self.visited_state_value) 
            print("Loaded Save Rtree, len:",self.visited_state_counter)
        obs_dimension = 16
        self.visited_state_tree_prop = rindex.Property()
        self.visited_state_tree_prop.dimension = obs_dimension+1
        # self.visited_state_dist = np.array([[1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5,1, 1, 0.5, 0.5,1, 1, 0.5, 0.5, 0.1]])#, 10, 0.3, 3, 1, 0.1]])
        self.visited_state_dist = np.array([[2, 2, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0,  0.5]])#, 10, 0.3, 3, 1, 0.1]])
        # self.visited_state_dist = np.array([[2, 2, 1.0, 1.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0,  0.5]])#, 10, 0.3, 3, 1, 0.1]])
        self.visited_state_tree = rindex.Index('state_index',properties=self.visited_state_tree_prop)

        self.visited_value_outfile = open("visited_value.txt", "a")
        self.visited_value_format = " ".join(("%f",)*2)+"\n"
    
    def update_with_replay_buffer(self, replay_buffer):
        print("Start Update Rtree!Len:",len(replay_buffer._storage))
        j=0
        for experience in replay_buffer._storage:
            obs_e, action_e, rew, new_obs, done, masks, train_times = experience
            for i in range(train_times):
                state_to_record = np.append(obs_e, action_e)
                self.visited_state_tree.insert(self.visited_state_counter,
                            tuple((state_to_record-self.visited_state_dist[0]).tolist()+(state_to_record+self.visited_state_dist[0]).tolist()))
                self.visited_state_counter += 1
            j += 1
            print("Updated count:",j)
        print("Rtree using Train Buffer Updated!,Len:",self.visited_state_counter)

    def add_data_to_rtree(self, training_data):
        (obses_t, actions, rewards, obses_tp1, dones, masks, weights, batch_idxes, training_time) = training_data
        for i in range(len(obses_t)):
            state_to_record = np.append(obses_t[i], actions[i])
            self.visited_state_tree.insert(self.visited_state_counter,
                        tuple((state_to_record-self.visited_state_dist[0]).tolist()+(state_to_record+self.visited_state_dist[0]).tolist()))

            self.visited_state_counter += 1
            self.visited_value_outfile.write(self.visited_value_format % tuple([actions[i],rewards[i]]))

    def calculate_visited_times(self, obs, action):
        
        state_to_count = np.append(obs, action)
        visited_times = sum(1 for _ in self.visited_state_tree.intersection(state_to_count.tolist()))

        return visited_times

class Bound(object):

    def __init__(self,
                 is_training = True,
                 debug = True,):

        self.is_training = is_training
        self.obs_dimension = 16  
        self.rule_var_thres = 0.5
        self.rl_var_thres = 0.5
        self.confidence_thres = 0.5
        self.rule_times = 0
        self.rl_times = 0
    
    def act_train(self, obs, q_values, bootstrap_action, rtree, num_iters):
        uncertainty_thres = 5
        var_thres = 0.05
        visited_time_thres = 30
        return self.act_train_bootstrap_optimal(obs, q_values, uncertainty_thres, var_thres, rtree, visited_time_thres, num_iters)
        # if self.should_use_rule(obs, q_values, bootstrap_action, rtree.calculate_visited_times(obs,0)): 
        #     return np.array(0)
        # else:
        #     # Deep exploration from Bootstrapped DQN
        #     return bootstrap_action

    def should_use_rule(self, obs, q_values, bootstrap_action, train_times):
        """
        Whether the state should use rule action
        """
        if bootstrap_action == 0:
            print("[Bootstrapped Dqn]: Running: Rule_action")
            return np.array(0)

        q_rule_list = []
        q_list = q_values(obs[None])
        for i in range(10):
            q_rule_list.append(q_list[i][0][0])

        mean_rule = np.mean(np.array(q_rule_list))
        var_rule = np.var(np.array(q_rule_list))
        # print("[Bootstrapped Dqn]: Qrule list:",q_rule_list)
        # print("[Bootstrapped Dqn]: Mean and Var:",mean_rule, var_rule)

        # temp
        q_rl_list = []
        meanRL_list = []
        varRL_list = []
        for candidate_action in range(0,8):
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])                
            meanRL_list.append(np.mean(np.array(q_rl_list)))
            varRL_list.append(np.var(np.array(q_rl_list)))

        # print("[Bootstrapped Dqn]: meanRL_list",meanRL_list)
        # print("[Bootstrapped Dqn]: varRL_list",varRL_list)

        if var_rule > 0.05 or train_times < 30: #var_thres
            return True

        # Rule perform good
        # mean_rule in (0,1), related to reward
        # explore_motivation = random.uniform(0,1)
        # if explore_motivation < (mean_rule + (1 - mean_rule) * var_rule / 100): # higher var,lower exploration
        #     return True
        print("[Bootstrapped Dqn]: RL exploration")
        return False

    def act_test_gaussion(self, obs, bootstrap_action, act_ubp, exploration, num_iters):
        if bootstrap_action == 0:
            print("[Bootstrapped Dqn]: Running: Rule_action")
            return np.array(0)

        q_rule_list = []
        q_rl_list = []
        for i in range(10):
            q_list = act_ubp(obs[None], head=i, update_eps=exploration.value(num_iters))[0]
            q_rule_list.append(q_list[0])
        mean_rule = np.mean(np.array(q_rule_list))
        var_rule = np.var(np.array(q_rule_list))
        print("[Bootstrapped Dqn]: Qrule list:",q_rule_list)
        print("[Bootstrapped Dqn]: Mean and Var:",mean_rule, var_rule)
        for candidate_action in range(1,16):
            for i in range(10):
                q_list = act_ubp(obs[None], head=i, update_eps=exploration.value(num_iters))[0]
                q_rl_list.append(q_list[candidate_action])
            mean_RL = np.mean(np.array(q_rl_list))
            var_RL = np.var(np.array(q_rl_list))
            
            if var_rule < self.rule_var_thres or mean_rule > 0:
                continue

            var_diff = var_rule + var_RL
            sigma_diff = np.sqrt(var_diff)
            mean_diff = mean_RL - mean_rule

            z = mean_diff/sigma_diff
            print(candidate_action, 1 - stats.norm.cdf(-z))
            if 1 - stats.norm.cdf(-z) > self.confidence_thres:
                print("[Bootstrapped Dqn]: RL take over! Action:",candidate_action)
                return np.array(candidate_action)
        
        print("[Bootstrapped Dqn]: Running: Rule_action")
        return np.array(0)

    def act_test_bootstrap_probability(self, obs, q_values, uncertainty_thres, var_thres, rtree, visited_time_thres):
        
        q_rule_list = []
        q_rl_list = []
        action_vote_list = np.zeros(8)

        q_list = q_values(obs[None])
        for i in range(10):
            q_rule_list.append(q_list[i][0][0])

        mean_rule = np.mean(np.array(q_rule_list))
        var_rule = np.var(np.array(q_rule_list))
        print("[Bootstrapped Dqn]: Qrule list:",q_rule_list)
        print("[Bootstrapped Dqn]: Mean and Var:",mean_rule, var_rule)
        
        # temp
        q_rl_list = []
        meanRL_list = []
        varRL_list = []
        visited_times_list = []

        for candidate_action in range(0,8):
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])                
            meanRL_list.append(np.mean(np.array(q_rl_list)))
            varRL_list.append(np.var(np.array(q_rl_list)))
            q_rl_list = []
            visited_times_list.append(rtree.calculate_visited_times(obs, candidate_action))

        print("[Bootstrapped Dqn]: meanRL_list",meanRL_list)
        print("[Bootstrapped Dqn]: varRL_list",varRL_list)
        print("[Bootstrapped Dqn]: visited_times_list",visited_times_list)

        if var_rule > var_thres or visited_times_list[0] < visited_time_thres:
            self.rule_times = self.rule_times + 1
            print("[Bootstrapped Dqn]: Running: Rule_action")
            return np.array(0)
        
        better = []

        for candidate_action in range(0,8):
            if_better = 0
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])
                if q_list[i][0][candidate_action] > q_list[i][0][0]:
                    if_better = if_better + 1
                
            better.append(if_better)

        pro_opti_action = np.where(better==np.max(better,axis=0))
        print("[Bootstrapped Dqn]: Better List",better)
        print("pro_opti_action",pro_opti_action)

        for rl_action in list(pro_opti_action[0]):
            # print("rl_action",rl_action)
            # print("rl_action",better[rl_action])
            # print("rl_action",meanRL_list[rl_action])
            # print("rl_action",varRL_list[rl_action])
            if better[rl_action] > uncertainty_thres and meanRL_list[rl_action] > mean_rule and varRL_list[rl_action] < var_thres and visited_times_list[rl_action]>visited_time_thres:
                print("[Bootstrapped Dqn]: RL take over! Action:",rl_action, better,  meanRL_list[rl_action] > mean_rule, varRL_list[rl_action])
                self.rl_times = self.rl_times + 1
                return np.array(pro_opti_action[0][0])
        
        print("[Bootstrapped Dqn]: Running: Rule_action")
        self.rule_times = self.rule_times + 1
        return np.array(0)

    def act_test_bootstrap_optimal(self, obs, q_values, uncertainty_thres, var_thres, rtree, visited_time_thres):

        q_rule_list = []
        q_rl_list = []
        action_vote_list = np.zeros(8)
        meanRL_list = []
        varRL_list = []
        visited_times_list = []

        q_list = q_values(obs[None])

        for candidate_action in range(0,8):
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])                
            meanRL_list.append(np.mean(np.array(q_rl_list)))
            varRL_list.append(np.var(np.array(q_rl_list)))
            q_rl_list = []
            visited_times_list.append(rtree.calculate_visited_times(obs, candidate_action))

        print("[Bootstrapped Dqn]: meanRL_list",meanRL_list)
        print("[Bootstrapped Dqn]: varRL_list",varRL_list)
        print("[Bootstrapped Dqn]: visited_times_list",visited_times_list)

        for i in range(10):
            q_rule_list.append(q_list[i][0][0])
            q_max_action = np.where(q_list[i][0]==np.max(q_list[i][0]))
            action_vote_list[q_max_action] = action_vote_list[q_max_action] + 1

        vote_action = np.where(action_vote_list==np.max(action_vote_list))
        mean_rule = np.mean(np.array(q_rule_list))
        var_rule = np.var(np.array(q_rule_list))
        print("[Bootstrapped Dqn]: Vote_action:",vote_action, action_vote_list)
        # print("[Bootstrapped Dqn]: Qrule list:",q_rule_list)
        # print("[Bootstrapped Dqn]: Mean and Var:",mean_rule, var_rule)


        if var_rule > var_thres or visited_times_list[0] < visited_time_thres: #var_thres
            print("[Bootstrapped Dqn]: Running: Rule_action_1")
            return np.array(0)

        for candidate_action in vote_action[0]:    
            better = 0
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])
                if q_list[i][0][candidate_action] > q_list[i][0][0]:
                    better = better + 1
            
            mean_RL = np.mean(np.array(q_rl_list))
            var_RL = np.var(np.array(q_rl_list))
            
            # print("[Bootstrapped Dqn]: Qrl list:",q_rl_list, candidate_action)
            q_rl_list = []

            if better > uncertainty_thres and mean_RL > mean_rule and rtree.calculate_visited_times(obs, candidate_action) > visited_time_thres and var_RL < var_thres:
                print("[Bootstrapped Dqn]: RL take over! Action:",candidate_action, better, mean_RL - mean_rule)
                self.rl_times = self.rl_times + 1

                return np.array(candidate_action)

        self.rule_times = self.rule_times + 1
        print("[Bootstrapped Dqn]: Running: Rule_action_2")
        return np.array(0)
        
    def act_train_bootstrap_optimal(self, obs, q_values, uncertainty_thres, var_thres, rtree, visited_time_thres, num_iters):

        explore_motivation = random.uniform(0,1) 
        if explore_motivation < 0.2 * (300000 - num_iters) / 300000: # higher var,lower exploration
            print("[Bootstrapped Dqn]: Running: Random_action")
            return np.array(random.randint(0,7))
        
        q_rule_list = []
        q_rl_list = []
        action_vote_list = np.zeros(8)
        meanRL_list = []
        varRL_list = []
        visited_times_list = []

        q_list = q_values(obs[None])

        for i in range(10):
            q_rule_list.append(q_list[i][0][0])
            q_max_action = np.where(q_list[i][0]==np.max(q_list[i][0]))
            action_vote_list[q_max_action] = action_vote_list[q_max_action] + 1

        vote_action = np.where(action_vote_list==np.max(action_vote_list))
        mean_rule = np.mean(np.array(q_rule_list))
        var_rule = np.var(np.array(q_rule_list))
        # print("[Bootstrapped Dqn]: Vote_action:",vote_action, action_vote_list)
        # print("[Bootstrapped Dqn]: Qrule list:",q_rule_list)
        # print("[Bootstrapped Dqn]: Mean and Var:",mean_rule, var_rule)

        for candidate_action in range(0,8):
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])                
            meanRL_list.append(np.mean(np.array(q_rl_list)))
            varRL_list.append(np.var(np.array(q_rl_list)))
            q_rl_list = []
            visited_times_list.append(rtree.calculate_visited_times(obs, candidate_action))

        # print("[Bootstrapped Dqn]: meanRL_list",meanRL_list)
        # print("[Bootstrapped Dqn]: varRL_list",varRL_list)
        # print("[Bootstrapped Dqn]: visited_times_list",visited_times_list)

        if var_rule > var_thres or visited_times_list[0] < visited_time_thres: #var_thres
            print("[Bootstrapped Dqn]: Running: Rule_action_1")
            return np.array(0)

        for candidate_action in vote_action[0]:    
            better = 0
            for i in range(10):
                q_rl_list.append(q_list[i][0][candidate_action])
                if q_list[i][0][candidate_action] > q_list[i][0][0]:
                    better = better + 1
            
            mean_RL = np.mean(np.array(q_rl_list))
            var_RL = np.var(np.array(q_rl_list))
            
            # print("[Bootstrapped Dqn]: Qrl list:",q_rl_list, candidate_action)
            q_rl_list = []

            if better > uncertainty_thres and mean_RL > mean_rule:
                print("[Bootstrapped Dqn]: RL take over! Action:",candidate_action, better, mean_RL - mean_rule)
                self.rl_times = self.rl_times + 1

                return np.array(candidate_action)

        self.rule_times = self.rule_times + 1
        print("[Bootstrapped Dqn]: Running: Rule_action_2")
        return np.array(0)

    def clean_running_times(self):
        self.rule_times = 0
        self.rl_times = 0





