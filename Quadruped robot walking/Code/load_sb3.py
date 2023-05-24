import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform

if platform == "darwin":  # mac
    import PyQt5

    matplotlib.use("Qt5Agg")
else:  # linux
    matplotlib.use('TkAgg')

# stable baselines
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.cmd_util import make_vec_env

from env.quadruped_gym_env import QuadrupedGymEnv
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results
from utils.datalogger import DATALOGGER

interm_dir = "./logs/intermediate_models/"
env_config = {}

# Models
## '122221185611' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_BASE , reward: FWD_ENERGY_FALL, env: obstacles, number iteration: 1.5M
## '122321171323' algorithm: SAC, action: CARTESIAN_PD,  observation: LR_BASE , reward: FWD_ENERGY_FALL, env: obstacles, number iteration: 1.5M
## '122621005954' algorithm: PPO, action: PD,  observation: LR_BASE , reward: FWD_ENERGY_FALL, env: obstacles, number iteration: 1.5M
## '122621105928' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_BASE_FEET , reward: FWD_ENERGY_FALL, env: obstacles, number iteration: 1.5M

## '122721084849' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_BASE , reward: FWD_ENERGY_FALL, env: flat, number iteration: 1.5M
## '122821060537' algorithm: SAC, action: CARTESIAN_PD,  observation: LR_BASE , reward: FWD_ENERGY_FALL, env: flat, number iteration: 1.5M
## '122921170401' algorithm: PPO, action: PD,  observation: LR_BASE , reward: FWD_ENERGY_FALL, env: flat, number iteration: 1.5M
## '122621224414' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_BASE_FEET , reward: FWD_ENERGY_FALL, env: flat, number iteration: 1.5M

## '010322170232' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_BASE , reward: FWD_FALL, env: flat, number iteration: 1.5M
## 'obs' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_ROBUST, reward: ROBUST_OBSTACLES, env: obstacles (domain randomisation), number iteration: 3M
## 'y_speed_match' algorithm: PPO, action: CARTESIAN_PD,  observation: LR_ROBUST, reward: ROBUST_MATCH, env: flat, number iteration: 600k


## Note that all the simualtion not used on the report are not listed here

log_dir = interm_dir + '122221185611'
LEARNING_ALG = "PPO"
env_config['motor_control_mode'] = "CARTESIAN_PD"  # CARTESIAN_PD or PD
env_config['observation_space_mode'] = "LR_BASE"  # DEFAULT or LR_BASE or LR_BASE_FEET or LR_BASE_FEET_VREF or LR_ROBUST
env_config['task_env'] = "FWD_ENERGY_FALL"  # FWD_LOCOMOTION or FWD_FALL or FWD_ENERGY_FALL or ROBUST_OBSTACLES or ROBUST_MATCH
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False
env_config['domain_randomisation'] = True
env_config['competition_env'] = False
PLOT = True
COT = True
DUTY = True
nb_step_evaluation = 3000
time_step = 0.001



# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir], 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show()

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False  # do not update stats at test time
env.norm_reward = False  # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0
logdata = DATALOGGER()

for i in range(nb_step_evaluation):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    if dones:
        print('episode_reward', episode_reward)
        episode_reward = 0
    print(i)
    logdata.log(rewards,
                base_pos=info[0]['base_pos'],
                base_orientation=info[0]['base_orientation'],
                base_linear_velocity=info[0]['base_linear_velocity'],
                base_angular_velocity=info[0]['base_angular_velocity'],
                motor_angle=info[0]['motor_angle'],
                motor_velocity=info[0]['motor_velocity'],
                motor_torque=info[0]['motor_torque'],
                feet_position=info[0]['feet_position'],
                feet_velocity=info[0]['feet_velocity'],
                feet_contact_bool=info[0]['feet_contact_bool'],
                time=info[0]['time'])

if PLOT:
    logdata.plot(start=None, stop=None)
if COT:
    logdata.CoT_simu(nb_step_evaluation)
if DUTY:
    logdata.duty_ratio(nb_step_evaluation)
