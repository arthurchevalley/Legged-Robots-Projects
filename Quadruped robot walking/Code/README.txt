# Quadruped-Sim
This repository contains an environment for simulating a quadruped robot.


## Installation
Recommend using a virtualenv (or conda) with python3.6 or higher. After installing virtualenv with pip, this can be done as follows:

`virtualenv {quad_env, or choose another name venv_name} --python=python3`

To activate the virtualenv: 

`source {PATH_TO_VENV}/bin/activate` 

Your command prompt should now look like: 

`(venv_name) user@pc:path$`

The repository depends on recent versions of pybullet, numpy, etc., which you can install inside your virtual environment with: 

`pip install -r requirements.txt `


## Requirements
gym==0.19.0
matplotlib==3.3.4
numpy==1.19.5
pybullet==3.2.0
stable-baselines3==1.3.0
pyqt5==5.15.6


## Code structure
- [env](./env) for the quadruped environment files, please see the gym simulation environment [quadruped_gym_env.py](./env/quadruped_gym_env.py), the robot specific functionalities in [quadruped.py](./env/quadruped.py), and config variables in [configs_a1.py](./env/configs_a1.py). You will need to make edits in [quadruped_gym_env.py](./env/quadruped_gym_env.py), and review [quadruped.py](./env/quadruped.py) carefully for accessing robot states and calling functions to solve inverse kinematics, return the leg Jacobian, etc. 
- [a1_description](./a1_description) contains the robot mesh files and urdf.
- [utils](./utils) for some file i/o and plotting helpers.
- [hopf_network.py](./hopf_polar.py) provides a CPG class skeleton for various gaits, and maps these to be executed on an instance of the  [quadruped_gym_env](./env/quadruped_gym_env.py) class. Please fill in this file carefully. 
- [hopf_network_gridSearch.py](./hopf_polar.py) provides a grid-search to tune the CPG parameters for various gaits, and maps these to be executed on an instance of the  [quadruped_gym_env](./env/quadruped_gym_env.py) class. Please fill in this file carefully. 
- [run_sb3.py](./run_sb3.py) and [load_sb3.py](./load_sb3.py) provide an interface to training RL algorithms based on [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). You should review the documentation carefully for information on the different algorithms and training hyperparameters. 


## Usage
# hopf_network 
Before running please take care of the following parameters:
- CPG parameters (Alpha, P Gains Joint, P Gains Cartesian, D Gains Joint, D Gains Cartesian - lines 33-42) and (Swing frequency, Stance Frequency,  gait, coefficient to multiply coupling matrix, couple boolean, foot swing height gc, foot stance penetration, robot height, desired step length - lines 84-103)
- PD-Controller contribution (Cartesian, Joint or both - lines 51-53)
- The number of simulation rounds - lines 55-56
- The plot boolean - lines 58-59
- The motion direction (1 = forward, 0 = backward) - line 61-62
- The motion direction (LATERAL_MOTION: 0 = moving along x, 1 = moving along y or DIAGONAL_MOTION to move along x-y) - lines 64-66
- The bolean VMC roll and pitch rejection of the base - lines 68-69
- The bolean  VMC orientation tracking of the base - lines 71-72

# hopf_network_gridSearch
Before running please take care of the following parameters:
- Initialsie the gait to tune - line 45
- Select a TUNE mode (TUNE=True test the combinasion of all the parameters or TUNE=False Apply the best combinasion saved in 'name_file') - 229-230
- The plot and print boolean - lines 232-233
- PD-Controller contribution (Cartesian, Joint or both - lines 235-237)
- The objective reward - line 239-242
- Combinaison of CPG parameters tested by the grid search - lines 247-259

# run_sb3
Before running please take care of the following parameters:
- Select the learning algorithm (between PPO and SAC) - lines 18
- Configurate the hyperparameters of the learning algorithm (PPO - lines 58-76, SAC - lines 78-92)
- Select the number of iteration - line 23
- Select the PD-Controller (Cartesian or Joint) - line 24
- Select the reward function - line 25
- Select the observation space mode - line 26
- Boolean for domain randomisation - line 27
- Boolean for adding noise - line 28
To load a model please put LOAD_NN (line 19) at true and complete the model name in line 20


# Note that to train the robust controller, self.training must be set to True in quadruped.py line 45
# Note that to test the robust controller, self.training must be set to False in quadruped.py line 45
# The desired matching speed must be specified on lines 474 and 475. The first nbr_iterations, the reference speed is [1.0,0.0]
# Finally, the number of training step per speed can be set on line 443 with variable "nbr_iter_compare"

# load_sb3
Before running please take care of the following parameters:
- Select the model to load - line 48
- Select the learning algorithm (between PPO and SAC) - lines 49
- Select the PD-Controller used for the model (Cartesian or Joint) - line 50
- Select the observation space mode - line 51
- Select the reward function - line 52
- Boolean to render - line 53
- Boolean to record a video - line 54
- Boolean for adding noise - line 55
- Boolean to test the environnement with random rectangular obstacles and random mass - line 56
- Boolean to test the competition environnement - line 57
- The plot boolean for graphes - lines 58
- Boolean for the CoT value - lines 59
- Boolean for the Duty factor value - lines 60
- Select the number of simulation step - lines 61



## Code resources
- The [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) is the current up-to-date documentation for interfacing with the simulation. 
- The quadruped environment took inspiration from [Google's motion-imitation repository](https://github.com/google-research/motion_imitation) based on [this paper](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf). 
- Reinforcement learning algorithms from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3). Also see for example [ray[rllib]](https://github.com/ray-project/ray) and [spinningup](https://github.com/openai/spinningup). 


## Conceptual resources
The CPGs are based on the following papers:
- L. Righetti and A. J. Ijspeert, "Pattern generators with sensory feedback for the control of quadruped locomotion," 2008 IEEE International Conference on Robotics and Automation, 2008, pp. 819-824, doi: 10.1109/ROBOT.2008.4543306. [link](https://ieeexplore.ieee.org/document/4543306)
- M. Ajallooeian, S. Pouya, A. Sproewitz and A. J. Ijspeert, "Central Pattern Generators augmented with virtual model control for quadruped rough terrain locomotion," 2013 IEEE International Conference on Robotics and Automation, 2013, pp. 3321-3328, doi: 10.1109/ICRA.2013.6631040. [link](https://ieeexplore.ieee.org/abstract/document/6631040) 
- M. Ajallooeian, S. Gay, A. Tuleu, A. Spröwitz and A. J. Ijspeert, "Modular control of limit cycle locomotion over unperceived rough terrain," 2013 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2013, pp. 3390-3397, doi: 10.1109/IROS.2013.6696839. [link](https://ieeexplore.ieee.org/abstract/document/6696839) 


## License
EPFL Legged Robotic MICRO-507