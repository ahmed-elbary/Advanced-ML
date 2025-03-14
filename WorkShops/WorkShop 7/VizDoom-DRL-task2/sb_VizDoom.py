#####################################################
# sb-VizDoom.py
# 
# This program trains Deep Reinforcement Learning (DRL) agents to learn to play 
# the games available in VizDoom. The VizDoom environments provide a first-person 
# shooter (FPS) simulation for training agents. The agent's goal is to interact with 
# the environment by taking/executing actions based on visual and game state information.
# It supports agent training with different algorithms like DQN, A2C, PPO, powered by
# stable-baselines3, and evaluating and visualising the agent's performance. 
# 
# Link to VizDoom: https://github.com/mwydmuch/ViZDoom
# Link to Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
#
# This program has also been tested with non-image based environments like the popular
# LunarLander: https://gymnasium.farama.org/environments/box2d/lunar_lander/
#  
# Depending on your environment setup, the key dependencies required are:
#   pip install vizdoom
#   pip install stable-baselines3
#   pip install opencv-python
# 
# Other dependencies to install, for the environments of this workshop, include:
#   apt-get update && apt-get install -y python3-opencv
#   apt-get install libgl1
#   pip install swig
#   pip install "gymnasium[box2d]"
#   pip install readchar
#
# and this if you see "This application failed to start because no Qt platform plugin could be initialized..."
# apt install qt6-base-dev
# 
# This program has also been tested in WSL compiling the code in the following link:
# git clone https://github.com/Farama-Foundation/ViZDoom.git
# 
# Some research publications related to VizDoom:
# https://arxiv.org/pdf/1605.02097
# https://arxiv.org/pdf/1809.03470
# https://www.diva-portal.org/smash/get/diva2:1679888/FULLTEXT01.pdf
# 
# Version 1.0 -- adapted for training and testing agents with MarioBros environments
# Version 2.0 -- rewritten for training and testing agents with VizDoom environments
# Version 2.1 -- revised for training and testing agents with non-VizDoom environments
#                such as "LunarLander-v3", which makes the program more general for DRL.
# Contact: hcuayahuitl@lincoln.ac.uk
#####################################################

import os
import sys
import cv2
import time
import pickle
import random
import numpy as np
import gymnasium 
import vizdoom.gymnasium_wrapper
from stable_baselines3 import DQN,A2C,PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor
from stable_baselines3.common.vec_env import DummyVecEnv


# Returns an observation containing a resised image and other info.
# frame_skip=the number of frames to skip between actions to speed up training
class ObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env, shape, frame_skip):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = frame_skip 

        # create new observation space with the new shape
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        if observation.shape[-1] != 3:  # if the channels are not 3 (corresponding to RGB)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) # convert to RGB24
        return observation

# Converts RGB observations to grayscale to reduce input dimensionality.
# gymnasium.spaces.Box() creates a continuous space for observations.
# see https://gymnasium.farama.org/main/_modules/gymnasium/spaces/box/
class GrayscaleObservationWrapper(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(
            0, 255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8
        ) # only 1 channel (grayscale) instead of 3 (RGB)

    def observation(self, observation):
        # check that the observation is in RGB format before converting to grayscale
        if observation.shape[-1] != 3:  # if the channels are not 3 (corresponding to RGB)
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB) # convert to RGB24

        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # convert to grayscale
        gray = np.expand_dims(gray, axis=-1) # expand dimensions to match input shape (H, W, 1)
        return gray


# Class for creating DRL agents through environment setup, model creation, training, evaluation, and policy rendering
class DRL_Agent:
    # initialise the DRL agent with the given parameters
    def __init__(self, environment_id, learning_alg, train_mode=True, seed=None, n_envs=8, frame_skip=4):
        self.environment_id = environment_id
        self.learning_alg = learning_alg
        self.train_mode = train_mode
        self.seed = seed if seed else random.randint(0, 1000)
        self.policy_filename = f"{learning_alg}-{environment_id}-seed{self.seed}.policy.pkl"
        self.n_envs = n_envs if train_mode else 1  # number of environments for training or 1 for testing
        self.frame_skip = frame_skip # number of frames to skip between actions in the environment
        self.image_shape = (84, 84) # image res (height, width):e.g., (240, 320); (120, 160); (60, 80); 
        self.training_timesteps = 10000 # total number of timesteps for training the agent
        self.num_test_episodes = 20 # number of episodes to run for testing the trained agent
        self.l_rate = 0.00083 # learning rate for the optimiser during training
        self.gamma = 0.995 # discount factor for future rewards (used in RL algorithms)
        self.n_steps = 512 # number of steps/actions the agent will take before updating the model
        self.policy_rendering = True # if True, shows visualisations of the learnt behaviour
        self.rendering_delay = 0.05 if self.environment_id.find("Vizdoom") > 0 else 0 # delay in rendering
        self.log_dir = './logs' # directory to store the logs containing agent performance
        self.model = None # initialises the model that will define the agent's policy & learning behavior
        self.policy = None # initialises the policy to "MlpPolicy" or "CnnPolicy" depending on the environment
        self.environment = None # initialises the environment as None, to be set later (gym environment)

        self._check_environment()
        self._create_log_directory()

    # check if the specified environment is available, stops execution otherwise
    def _check_environment(self):
        available_envsA = [env for env in gymnasium.envs.registry.keys() if "LunarLander" in env]
        available_envsB = [env for env in gymnasium.envs.registry.keys() if "Vizdoom" in env]
        if self.environment_id in available_envsA :
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsA}")
        elif self.environment_id in available_envsB:
            print(f"ENVIRONMENT_ID={self.environment_id} is available in {available_envsB}")
        else:
            print(f"UNKNOWN environment={self.environment_id}")
            print(f"AVAILABLE_ENVS={available_envsA, available_envsB}")
            sys.exit(0)

    # creates a log directory if it doesn't exist already
    def _create_log_directory(self):
        #if self.environment_id.find("Vizdoom") == -1: return
        # Only logs info for Vizdoom environments and uses a single folder -- be careful with that.
        # You can/should rename the logs folder with the appropriate seed number for your own records.
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir) 
            print(f"Log directory created: {self.log_dir}")
        else:
            print(f"Log directory {self.log_dir} already exists!")
    
    # wrapper function to customise environments for image-based obsertations
    def wrap_env(self, env):
        env = ObservationWrapper(env, shape=self.image_shape, frame_skip=self.frame_skip)
        env = GrayscaleObservationWrapper(env) # convert to grayscale
        if self.train_mode: 
            # scale rewards for training stability, only during training
            env = gymnasium.wrappers.TransformReward(env, lambda r: r * 0.01) 
        return env

    # Create the vectorised environment with multiple parallel environments.
	# Since seeding the environment will use the same initial game state each time 
	# training starts, you should train your agent on multiple different seeds. Then, 
	# report mean and standard deviation of performance metrics across those seeds.
    def create_environment(self, use_rendering=False):
        print("self.environment_id="+str(self.environment_id))

        if self.environment_id.find("Vizdoom") == -1:
            if use_rendering: 
                self.environment = gymnasium.make(self.environment_id, render_mode="human")
            else:
                self.environment = gymnasium.make(self.environment_id)
            self.environment = DummyVecEnv([lambda: self.environment])
            self.environment = VecMonitor(self.environment, self.log_dir) 
            self.policy = "MlpPolicy"
        
        else:
            self.environment = make_vec_env(
                self.environment_id,
                n_envs=self.n_envs,
                seed=self.seed,
                monitor_dir=self.log_dir,
                wrapper_class=self.wrap_env  # applies wrappers inside this function
            )
            self.environment = VecFrameStack(self.environment, n_stack=4)  # stacks frames for temporal context
            self.environment = VecTransposeImage(self.environment) # transposes image for correct format (channel first)
            self.policy = "CnnPolicy"

        print("self.environment.action_space:",self.environment.action_space)

    # create the RL model based on the chosen learning algorithm
    def create_model(self):
        if self.learning_alg == "DQN":
            self.model = DQN(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma, buffer_size=10000, batch_size=64, exploration_fraction=0.9, verbose=1)
			
        elif self.learning_alg == "A2C":
            self.model = A2C(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma, verbose=1)
			
        elif self.learning_alg == "PPO":
            self.model = PPO(self.policy, self.environment, seed=self.seed, learning_rate=self.l_rate, gamma=self.gamma, verbose=1)
			
        else:
            print(f"Unknown LEARNING_ALG={self.learning_alg}")
            sys.exit(0)

    # train the agent's model or load a pre-trained model from a file
    def train_or_load_model(self):
        print(self.model)
        if self.train_mode:
            self.model.learn(total_timesteps=self.training_timesteps)
            print(f"Saving policy {self.policy_filename}")
            pickle.dump(self.model.policy, open(self.policy_filename, 'wb'))
        else:
            print("Loading policy...")
            with open(self.policy_filename, "rb") as f:
                policy = pickle.load(f)
            self.model.policy = policy

    # evaluate the policy on the environment by running a number of test episodes
    def evaluate_policy(self):
        print("Evaluating policy...")
        mean_reward, std_reward = evaluate_policy(self.model, self.model.get_env(), n_eval_episodes=self.num_test_episodes)
        print(f"EVALUATION: mean_reward={mean_reward} std_reward={std_reward}")

    # render the agent's behavior in the environment and track cumulative reward
    def render_policy(self):
        steps_per_episode = 0
        reward_per_episode = 0
        total_cummulative_reward = 0
        episode = 1
        self.create_environment(True)
        env = self.environment
        obs = env.reset()

        print("DEMONSTRATION EPISODES:")
        while True:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps_per_episode += 1
            reward_per_episode += reward
            if any(done):
                print(f"episode={episode}, steps_per_episode={steps_per_episode}, reward_per_episode={reward_per_episode}")
                total_cummulative_reward += reward_per_episode
                steps_per_episode = 0
                reward_per_episode = 0
                episode += 1
                obs = env.reset()
            if self.policy_rendering:
                env.render("human")
                time.sleep(self.rendering_delay)
            if episode > self.num_test_episodes:
                print(f"total_cummulative_reward={total_cummulative_reward} avg_cummulative_reward={total_cummulative_reward / self.num_test_episodes}")
                break
        env.close()

    # main method to run the DRL agent
    def run(self):
        self.create_environment()
        self.create_model()
        self.train_or_load_model()
        self.evaluate_policy()
        self.render_policy()


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("USAGE: sb-VizDoom.py (train|test) (DQN|A2C|PPO) [seed_number]")
        print("EXAMPLE1: sb-VizDoom.py train PPO")
        print("EXAMPLE2: sb-VizDoom.py test PPO 476")
        sys.exit(0)

    environment_id = "LunarLander-v3" # simpler environment (no images, simpler observations) 
    #environment_id = "VizdoomTakeCover-v0" # default environment (image observations)
    train_mode = sys.argv[1] == 'train' # boolean parameter train/test
    learning_alg = sys.argv[2] # argument to communicate the algorithm to use
    seed = random.randint(0, 1000) if train_mode else int(sys.argv[3]) # random or predefined seed
    
    agent = DRL_Agent(environment_id, learning_alg, train_mode, seed)
    agent.run()