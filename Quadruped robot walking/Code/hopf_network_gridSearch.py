"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import numpy as np
from numpy import unravel_index
import matplotlib
import math
from sys import platform
if platform =="darwin": # mac
  import PyQt5
  matplotlib.use("Qt5Agg")
else: # linux
  matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv

#from env.quadruped import Quadruped
TOTAL_SIM_TIME = 10
STANCE = 0
SWING = 1
MASS = 12
GRAVITY = 9.81

# define motion direction (1 = forward, 0 = backward)
FORWARD_MOTION = 1

# define motion direction (0 = moving along x, 1 = moving along y)
LATERAL_MOTION = 0

class HopfNetwork():
  """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.  

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """
  def __init__(self,
                mu=1**2,                # converge to sqrt(mu)
                omega_swing=8*2*np.pi,  # swing omgea                                                                      TROT 8*2*np.pi || PACE 4*2*np.pi  || BOUND 4*2*np.pi || WALK 4*2*np.pi
                omega_stance=2*2*np.pi, # stance omega                                                                     TROT 2*2*np.pi || PACE 2*2*np.pi  || BOUND 1*2*np.pi || WALK 4*2*np.pi
                gait="PACE",            # change depending on desired gait
                coupling_strength=1,    # coefficient to multiply coupling matrix --> everyone set to 1 TROT 2*2*np.pi    TROT 1 || PACE 1 || BOUND 1.5 || WALK
                couple=True,            # should couple
                time_step=0.001,        # time step 
                ground_clearance=0.05,  # foot swing height gc 0.05                                                       TROT 0.05 || PACE 0.06 || BOUND 0.075 || WALK
                ground_penetration=0.01,# foot stance penetration into ground gp 0.01                                     TROT 0.01 || PACE 0.01 || BOUND 0.02 || WALK
                robot_height=0.27,      # in nominal case (standing)  0.25                                                TROT 0.25 || PACE 0.25 || BOUND 0.24 || WALK
                des_step_len=0.03,      # desired step length 0.04                                                        TROT 0.08 || PACE 0.04 || BOUND 0.06 || WALK
                ):
    
    ###############
    # initialize CPG data structures: amplitude is row 0, and phase is row 1
    self.X = np.zeros((2,4))

    # save parameters
    self.gait = gait
    self._alpha = 50
    self._mu = mu
    self._omega_swing = omega_swing
    self._omega_stance = omega_stance  
    self._couple = couple
    self._coupling_strength = coupling_strength
    self._dt = time_step
    self._set_gait(gait)
    
    #me to plot the change between stance and swing phase
    self.omega_FSM = np.zeros(4)
    
    #me to plot r_dot and theta_dot for each leg
    self.X_dot = np.zeros((2,4))

    # set oscillator initial conditions  
    self.X[0,:] = np.random.rand(4) * .1
    self.X[1,:] = self.PHI[0,:] 

    # save body and foot shaping
    self._ground_clearance = ground_clearance 
    self._ground_penetration = ground_penetration
    self._robot_height = robot_height 
    self._des_step_len = des_step_len
    self._last_base_position = [0, 0, 0]


  def _set_gait(self,gait):
    """ For coupling oscillators in phase space.
    """
    self.PHI_trot = np.array([[0, np.pi, np.pi, 0],[-np.pi, 0, 0, -np.pi], [-np.pi, 0, 0, -np.pi], [0, np.pi, np.pi, 0]])
    self.PHI_bound = np.array([[0, 0, -np.pi, -np.pi],[0, 0, -np.pi, -np.pi], [np.pi, np.pi, 0, 0], [np.pi, np.pi, 0, 0]])
    self.PHI_pace = np.array([[0, np.pi, 0, np.pi],[-np.pi, 0, -np.pi, 0], [0, np.pi, 0, np.pi], [-np.pi, 0, -np.pi, 0]])
    #self.PHI_walk = np.array([[0, np.pi, 3 * np.pi / 2, np.pi / 2], [np.pi, 0, np.pi / 2, 3 * np.pi / 2], [np.pi / 2, 3 * np.pi / 2, 0, np.pi],
    #   [3 * np.pi / 2, np.pi / 2, np.pi, 0]])
    self.PHI_walk = np.array(
      [[0, np.pi, 3*np.pi/2, np.pi/2], [-np.pi, 0, np.pi/2, -np.pi/2], [-3*np.pi/2, -np.pi/2, 0, -np.pi],  [-np.pi/2, np.pi/2, np.pi, 0]])
    self.PHI_pronk = np.array([[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    self.PHI_gallop = np.array([[0, -np.pi/5, -6*np.pi/5, -np.pi], [np.pi/5, 0, -np.pi, -4*np.pi/5], [6*np.pi/5, np.pi, 0, np.pi/5], [np.pi, 4*np.pi/5, -np.pi/5, 0]])

    if gait == "TROT":
      print('-------------------------------------------------------')
      print('TROT')
      self.PHI = self.PHI_trot
    elif gait == "PACE":
      print('PACE')
      self.PHI = self.PHI_pace
    elif gait == "BOUND":
      print('BOUND')
      self.PHI = self.PHI_bound
    elif gait == "WALK":
      print('WALK')
      self.PHI = self.PHI_walk
    elif gait == "PRONK":
      print('PRONK')
      self.PHI = self.PHI_pronk
    elif gait == "GALLOP":
      print('GALLOP')
      self.PHI = self.PHI_gallop
    else:
      raise ValueError( gait + 'not implemented.')


  def update(self):
    """ Update oscillator states. """
    
    # update parameters, integrate
    self._integrate_hopf_equations()
    
    x = np.zeros(len(self.X[0,:]))
    z = np.zeros(len(self.X[0,:]))
    # map CPG variables to Cartesian foot xz positions (Equations 8, 9)
    for i in range(len(self.X[0,:])):

        if FORWARD_MOTION:
            x[i] = - self._des_step_len*self.X[0,i]*np.cos(self.X[1,i]) # forward motion
        else:
            x[i] = self._des_step_len * self.X[0, i] * np.cos(self.X[1, i])  # backward motion


        if np.sin(self.X[1,i])>0:
            z[i] = - self._robot_height + self._ground_clearance*np.sin(self.X[1,i]) # [TODO]
        else:
            z[i] = - self._robot_height + self._ground_penetration*np.sin(self.X[1,i])

    return x, z


        
  def _integrate_hopf_equations(self):
    """ Hopf polar equations and integration. Use equations 6 and 7. """
    # bookkeeping - save copies of current CPG states
    nb_oscillators = len(self.X[0,:])
    X = self.X.copy()
    X_dot = np.zeros((2,4))

    # loop through each leg's oscillator
    for i in range(nb_oscillators):
      # get r_i, theta_i from X
      r, theta = X[0,i], X[1,i]
      # compute r_dot (Equation 6)
      r_dot = self._alpha*(self._mu - r**2)*r
      # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
      if np.sin(theta%(2*np.pi))>0:
        theta_dot = self._omega_swing
      else:
        theta_dot = self._omega_stance

      #to plot
      self.omega_FSM[i] = theta_dot
      
      # loop through other oscillators to add coupling (Equation 7)
      if self._couple:
        for j in range(nb_oscillators):
            # ATTENTION TO CHECK
            theta_dot += X[0,j] * self._coupling_strength * np.sin(X[1,j] - X[1,i] - self.PHI[i,j])
            # ATTENTION TO CHECK
            
      # set X_dot[:,i]
      self.X_dot[:,i] = [r_dot, theta_dot]

    # integrate 
    self.X += self.X_dot*self._dt
    
    # mod phase variables to keep between 0 and 2pi
    self.X[1,:] = self.X[1,:] % (2*np.pi)

def duty_cycle_and_ratio(feetInContactBool): #feetInContactBool = list of 1 and 0 for a single leg
  """ Computes the duty cycle (T_stance + T_swing) and the duty ratio D = T_stance / T_swing"""

  T_stance_list = []
  T_swing_list = []
  duty_cycle_list = []
  duty_ratio_list = []

  T_stance = 0
  T_swing = 0
  i = 0

  while i < len(feetInContactBool) :

    if feetInContactBool[i] == STANCE:
      T_stance = 0
      while i < len(feetInContactBool) and feetInContactBool[i] != SWING :
        T_stance = T_stance + TIME_STEP
        i += 1

      T_stance_list.append(T_stance)
    else:
      T_swing = 0
      while i < len(feetInContactBool) and feetInContactBool[i] != STANCE:
        T_swing = T_swing + TIME_STEP
        i += 1

      T_swing_list.append(T_swing)

  for j in range(min(len(T_swing_list), len(T_stance_list))):
    duty_cycle_list.append(T_swing_list[j] + T_stance_list[j])
    duty_ratio_list.append(T_stance_list[j] / T_swing_list[j])

  return duty_cycle_list, duty_ratio_list


if __name__ == "__main__":
  TIME_STEP = 0.001
  foot_y = 0.0838  # this is the hip length
  sideSign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)

  # TUNE Settings (TUNE=True test the combinasion of all the parameters or TUNE=False Apply the best combinasion saved in 'name_file')
  TUNE=True

  # Boolean for prints and plots
  VERBOSE = True

  # PD-Controller contribution
  ADD_CARTESIAN_PD = True
  ADD_JOINT_PD = True

  # Reward objective function of the grid search
  feetContactRewardWeight = 1 / 1000
  velocityRewardWeight = 50
  forwardRewardWeight = 50

  # Tuned parameters file
  name_file='Tune_Walk.npy' # Save the best results

  # Combinaison of parameters tested
  omega_swing=[16*np.pi]
  omega_stance=[4*np.pi]
  ground_clearance=[0.04, 0.05, 0.06]
  ground_penetration=[0.01, 0.2]
  robot_height=[0.23, 0.24, 0.25, 0.26, 0.27, 0.28]
  des_step_len=[0.02, 0.03, 0.04]
  coupling_strength=[0.8, 1, 1.2]
  convergence_limit_cycle=[40, 50, 60]
  joint_PD_gains1=[140, 150, 160]
  cartesian_PD_gains1=[2400, 2500, 2600]
  joint_PD_gains2=[1.5, 2, 2.5]
  cartesian_PD_gains2=[36, 40, 44]

  # Initialisation of the matrix for the results of each combinaison
  Result = np.zeros((len(omega_swing), len(omega_stance), len(ground_clearance), len(ground_penetration), len(robot_height),
                     len(des_step_len), len(coupling_strength), len(convergence_limit_cycle),
                     len(joint_PD_gains1), len(cartesian_PD_gains1), len(joint_PD_gains2), len(cartesian_PD_gains2)))

  # GRID SEARCH -----------------------------------------------------------------------------
  if (TUNE):
    env = QuadrupedGymEnv(render=False,  # visualize
                          on_rack=False,  # useful for debugging!
                          isRLGymInterface=False,  # not using RL
                          time_step=TIME_STEP,
                          action_repeat=1,
                          motor_control_mode="TORQUE",
                          add_noise=False,  # start in ideal conditions
                          # record_video=True
                          )
    for omega_swing_i in range(0, len(omega_swing)):
      for omega_stance_i in range(0, len(omega_stance)):
        for ground_clearance_i in range(0, len(ground_clearance)):
          for ground_penetration_i in range(0, len(ground_penetration)):
            for robot_height_i in range(0, len(robot_height)):
              for des_step_len_i in range(0, len(des_step_len)):
                for coupling_strength_i in range(0, len(coupling_strength)):
                  for convergence_limit_cycle_i in range(0, len(convergence_limit_cycle)):
                    for joint_PD_gains1_i in range(0, len(joint_PD_gains1)):
                      for cartesian_PD_gains1_i in range(0, len(cartesian_PD_gains1)):
                        for joint_PD_gains2_i in range(0, len(joint_PD_gains2)):
                          for cartesian_PD_gains2_i in range(0, len(cartesian_PD_gains2)):
                            # initialize Hopf Network, supply gait
                            cpg = HopfNetwork(time_step=TIME_STEP)
                            cpg._omega_swing = omega_swing[omega_swing_i]
                            cpg._omega_stance = omega_stance[omega_stance_i]
                            cpg._ground_clearance = ground_clearance[ground_clearance_i]
                            cpg._ground_penetration = ground_penetration[ground_penetration_i]
                            cpg._robot_height = robot_height[robot_height_i]
                            cpg._des_step_len = des_step_len[des_step_len_i]
                            cpg._coupling_strength = coupling_strength[coupling_strength_i]
                            cpg._alpha=convergence_limit_cycle[convergence_limit_cycle_i]

                            # joint PD gains
                            kp = np.array([joint_PD_gains1[joint_PD_gains1_i], joint_PD_gains1[joint_PD_gains1_i]/2, joint_PD_gains1[joint_PD_gains1_i]/2]) * 2  # np.array([150, 70, 70]) * 2
                            kd = np.array([joint_PD_gains2[joint_PD_gains2_i], joint_PD_gains2[joint_PD_gains2_i]/4, joint_PD_gains2[joint_PD_gains2_i]/4])       # np.array([2, 0.5, 0.5]) * 2

                            # Cartesian PD gains
                            kpCartesian = np.diag([cartesian_PD_gains1_i] * 3) * 2    #np.diag([2500] * 3) * 2
                            kdCartesian = np.diag([cartesian_PD_gains2_i] * 3)        #np.diag([40] * 3)

                            TEST_STEPS = int(TOTAL_SIM_TIME / (TIME_STEP)) #int(10 / (TIME_STEP))
                            t = np.arange(TEST_STEPS)*TIME_STEP

                            ############## Sample Gains ##############
                            #get initial position of the base
                            initial_pos = env.robot.GetBasePosition()

                            if cpg.gait == "TROT":
                              cmp_leg03 = []
                              cmp_leg12 = []
                            elif cpg.gait == "PACE":
                              cmp_leg02 = []
                              cmp_leg13 = []
                            elif cpg.gait == "BOUND":
                              cmp_leg01 = []
                              cmp_leg23 = []
                            elif cpg.gait == "WALK":
                              xyz=0
                            elif cpg.gait == "PRONK":
                              cmp_leg0123 = []
                            elif cpg.gait == "GALLOP":
                              xyz=0
                            else:
                              raise ValueError(cpg.gait + 'not implemented.')

                            CoT = 0
                            falled = 1 # not fall or jumped
                            for j in range(TEST_STEPS):
                              # initialize torque array to send to motors
                              action = np.zeros(12)
                              # get desired foot positions from CPG
                              xs,zs = cpg.update()
                              q = env.robot.GetMotorAngles()
                              dq = env.robot.GetMotorVelocities()
                              joint_torques = env.robot.GetMotorTorques()

                              for i in range(len(joint_torques)):
                                CoT += abs(dq[i] * joint_torques[i])

                              # loop through desired foot positions and calculate torques
                              for i in range(4):
                                # initialize torques for leg_i
                                tau = np.zeros(3)
                                # get desired foot i pos (xi, yi, zi) in leg frame
                                if LATERAL_MOTION:
                                  leg_xyz = np.array([0, sideSign[i] * foot_y + xs[i], zs[i]])  # motion selon y
                                else:
                                  leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])  # motion selon x
                                #leg_xyz = np.array([xs[i], sideSign[i] * foot_y + xs[i], zs[i]]) #motion selon x-y

                                # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
                                J, pos = env.robot.ComputeJacobianAndPosition(i)
                                # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
                                leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)  # [TODO

                                # add joint PD contribution
                                if ADD_JOINT_PD:
                                  # Add joint PD contribution to tau for leg i (Equation 4)
                                  dq_d = np.zeros(3)
                                  tau += kp * (leg_q - q[i*3:(i*3+3)]) + kd*(dq_d - dq[i*3:(i*3+3)])# [TODO]

                                # add Cartesian PD contribution
                                if ADD_CARTESIAN_PD:
                                  # Get current foot velocity in leg frame (Equation 2)
                                  v = J @ dq[i*3:(i*3+3)]
                                  # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
                                  v_d = np.zeros(3)
                                  tau += J.transpose() @ (kpCartesian @ (leg_xyz - pos) + kdCartesian @ (v_d - v))# [TODO]

                                # Set tau for legi in action vector
                                action[3*i:3*i+3] = tau
                              if cpg.gait == "TROT":
                                numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
                                cmp_leg03.append((feetInContactBool[0] == feetInContactBool[3]))
                                cmp_leg12.append((feetInContactBool[1] == feetInContactBool[2]))
                                feetContactReward=(sum(cmp_leg03)+sum(cmp_leg12))
                              elif cpg.gait == "PACE":
                                numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
                                cmp_leg02.append((feetInContactBool[0] == feetInContactBool[2]))
                                cmp_leg13.append((feetInContactBool[1] == feetInContactBool[3]))
                                feetContactReward = (sum(cmp_leg02) + sum(cmp_leg13))
                              elif cpg.gait == "BOUND":
                                numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
                                cmp_leg01.append((feetInContactBool[0] == feetInContactBool[1]))
                                cmp_leg23.append((feetInContactBool[2] == feetInContactBool[3]))
                                feetContactReward = (sum(cmp_leg01) + sum(cmp_leg23))
                              elif cpg.gait == "WALK":
                                xyz=0
                              elif cpg.gait == "PRONK":
                                numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
                                cmp_leg0123.append((feetInContactBool[0] == feetInContactBool[1] == feetInContactBool[2] == feetInContactBool[3]))
                                feetContactReward = sum(cmp_leg0123)
                              elif cpg.gait == "GALLOP":
                                xyz=0
                              else:
                                raise ValueError(cpg.gait + 'not implemented.')

                              base_pos = env.robot.GetBasePosition() # put Results to zero if base jump or fall
                              if (env.is_fallen() or (base_pos[2] > 0.45)) and not (cpg.gait == "PRONK"):
                                falled = 0
                              elif (env.is_fallen() or (base_pos[2] > 0.65)) and (cpg.gait == "PRONK"):
                                falled = 0

                              # send torques to robot and simulate TIME_STEP seconds
                              env.step(action)

                            # get final position of the base
                            final_pos = env.robot.GetBasePosition()

                            # REWARD Grid Search
                            current_base_position = env.robot.GetBasePosition()  # 3
                            current_joint_velocity = env.robot.GetMotorVelocities()  # 12
                            forward_reward = current_base_position[0] -  cpg._last_base_position[0]
                            cpg._last_base_position = current_base_position
                            base_velocity = math.sqrt((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2 + (final_pos[2] - initial_pos[2])**2) / t[-1]
                            print('Velocity of the base:', base_velocity, ' m/s')
                            print('Is falled :', not falled)
                            if cpg.gait == "TROT" or cpg.gait == "PACE" or cpg.gait == "BOUND" or cpg.gait == "PRONK":
                              velocityReward = base_velocity
                              finalReward=velocityReward+feetContactReward*feetContactRewardWeight
                              print('velocityReward :', velocityReward, '  and  feetContactReward :', feetContactReward*feetContactRewardWeight)
                            else:
                              velocityReward = base_velocity
                              finalReward=velocityReward*velocityRewardWeight
                            print('forward_reward :', forward_reward*forwardRewardWeight)
                            finalReward=finalReward+forward_reward*forwardRewardWeight
                            print('finalReward :', finalReward*falled)
                            Result[omega_swing_i][omega_stance_i][ground_clearance_i][ground_penetration_i][robot_height_i][des_step_len_i][coupling_strength_i][convergence_limit_cycle_i][joint_PD_gains1_i][cartesian_PD_gains1_i][joint_PD_gains2_i][cartesian_PD_gains2_i] = finalReward*falled
                            env.reset()

    # save results
    with open(name_file, 'wb') as f:
      np.save(f, Result)
    print("Successfully saved results")

# TEST --------------------------------------------------------------------------------
  else:
    env = QuadrupedGymEnv(render=True,  # visualize
                          on_rack=False,  # useful for debugging!
                          isRLGymInterface=False,  # not using RL
                          time_step=TIME_STEP,
                          action_repeat=1,
                          motor_control_mode="TORQUE",
                          add_noise=False,  # start in ideal conditions
                          # record_video=True
                          )
    with open(name_file, 'rb') as f:
      a = np.load(f)
    maxindex = np.argmax(a)
    max_index_dec = unravel_index(a.argmax(), a.shape)
    print("index best solution :", max_index_dec)
    print("Reward :", a[max_index_dec])

    # initialize Hopf Network, supply gait
    cpg = HopfNetwork(time_step=TIME_STEP)
    cpg._ground_clearance = omega_swing[max_index_dec[0]]
    cpg._ground_penetration = omega_stance[max_index_dec[1]]
    cpg._ground_clearance = ground_clearance[max_index_dec[2]]
    cpg._ground_penetration = ground_penetration[max_index_dec[3]]
    cpg._robot_height = robot_height[max_index_dec[4]]
    cpg._des_step_len = des_step_len[max_index_dec[5]]
    cpg._coupling_strength = coupling_strength[max_index_dec[6]]
    cpg._alpha = convergence_limit_cycle[max_index_dec[7]]

    # joint PD gains
    kp = np.array([joint_PD_gains1[max_index_dec[8]], joint_PD_gains1[max_index_dec[8]] / 2,
                   joint_PD_gains1[max_index_dec[8]] / 2]) * 2  # np.array([150, 70, 70]) * 2
    kd = np.array([joint_PD_gains2[max_index_dec[10]], joint_PD_gains2[max_index_dec[10]] / 4,
                   joint_PD_gains2[max_index_dec[10]] / 4])  # np.array([2, 0.5, 0.5]) * 2

    # Cartesian PD gains
    kpCartesian = np.diag([max_index_dec[9]] * 3) * 2  # np.diag([2500] * 3) * 2
    kdCartesian = np.diag([max_index_dec[11]] * 3)  # np.diag([40] * 3)

    TEST_STEPS = int(TOTAL_SIM_TIME / (TIME_STEP))  # int(10 / (TIME_STEP))
    t = np.arange(TEST_STEPS) * TIME_STEP

    # Save CPG
    if cpg.gait == "TROT":
      cmp_leg03 = []
      cmp_leg12 = []
    elif cpg.gait == "PACE":
      cmp_leg02 = []
      cmp_leg13 = []
    elif cpg.gait == "BOUND":
      cmp_leg01 = []
      cmp_leg23 = []
    elif cpg.gait == "WALK":
      xyz=0
    elif cpg.gait == "PRONK":
      cmp_leg0123 = []
    elif cpg.gait == "GALLOP":
      xyz = 0
    else:
      raise ValueError(cpg.gait + 'not implemented.')

    if VERBOSE:
      CPG_save_r1 = []
      CPG_save_r2 = []
      CPG_save_r3 = []
      CPG_save_r4 = []

      CPG_save_theta1 = []
      CPG_save_theta2 = []
      CPG_save_theta3 = []
      CPG_save_theta4 = []

      CPG_save_r1_dot = []
      CPG_save_r2_dot = []
      CPG_save_r3_dot = []
      CPG_save_r4_dot = []

      CPG_save_theta1_dot = []
      CPG_save_theta2_dot = []
      CPG_save_theta3_dot = []
      CPG_save_theta4_dot = []

      omega1_FSM_save = []
      omega2_FSM_save = []
      omega3_FSM_save = []
      omega4_FSM_save = []

      # Plot 2: desired VS actual foot position for leg 1
      leg0_desired_foot_pos_x = []
      leg0_desired_foot_pos_y = []
      leg0_desired_foot_pos_z = []

      leg0_actual_foot_pos_x = []
      leg0_actual_foot_pos_y = []
      leg0_actual_foot_pos_z = []

      # Plot 3: desired VS actual joint angles for leg 1
      leg0_desired_angle_hip = []
      leg0_desired_angle_thigh = []
      leg0_desired_angle_calf = []

      leg0_actual_angle_hip = []
      leg0_actual_angle_thigh = []
      leg0_actual_angle_calf = []

      feetInContactBool_save_leg0 = []
      feetInContactBool_save_leg1 = []
      feetInContactBool_save_leg2 = []
      feetInContactBool_save_leg3 = []

      # duty cycles and ratios
      duty_cycle_list_leg0 = []
      duty_cycle_list_leg1 = []
      duty_cycle_list_leg2 = []
      duty_cycle_list_leg3 = []

      duty_ratio_list_leg0 = []
      duty_ratio_list_leg1 = []
      duty_ratio_list_leg2 = []
      duty_ratio_list_leg3 = []

    ############## Sample Gains ##############

    # get initial position of the base
    initial_pos = env.robot.GetBasePosition()

    CoT = 0
    for j in range(TEST_STEPS):

      if VERBOSE:
        # Save CPG
        CPG_save_r1.append(cpg.X[0, 0].copy())
        CPG_save_r2.append(cpg.X[0, 1].copy())
        CPG_save_r3.append(cpg.X[0, 2].copy())
        CPG_save_r4.append(cpg.X[0, 3].copy())

        CPG_save_theta1.append(cpg.X[1, 0].copy())
        CPG_save_theta2.append(cpg.X[1, 1].copy())
        CPG_save_theta3.append(cpg.X[1, 2].copy())
        CPG_save_theta4.append(cpg.X[1, 3].copy())

        CPG_save_r1_dot.append(cpg.X_dot[0, 0].copy())
        CPG_save_r2_dot.append(cpg.X_dot[0, 1].copy())
        CPG_save_r3_dot.append(cpg.X_dot[0, 2].copy())
        CPG_save_r4_dot.append(cpg.X_dot[0, 3].copy())

        CPG_save_theta1_dot.append(cpg.X_dot[1, 0].copy())
        CPG_save_theta2_dot.append(cpg.X_dot[1, 1].copy())
        CPG_save_theta3_dot.append(cpg.X_dot[1, 2].copy())
        CPG_save_theta4_dot.append(cpg.X_dot[1, 3].copy())

        omega1_FSM_save.append(cpg.omega_FSM[0].copy())
        omega2_FSM_save.append(cpg.omega_FSM[1].copy())
        omega3_FSM_save.append(cpg.omega_FSM[2].copy())
        omega4_FSM_save.append(cpg.omega_FSM[3].copy())

      # initialize torque array to send to motors
      action = np.zeros(12)
      # get desired foot positions from CPG
      xs, zs = cpg.update()
      q = env.robot.GetMotorAngles()
      dq = env.robot.GetMotorVelocities()
      joint_torques = env.robot.GetMotorTorques()

      for i in range(len(joint_torques)):
        CoT += abs(dq[i] * joint_torques[i])

      # loop through desired foot positions and calculate torques
      for i in range(4):
        # initialize torques for leg_i
        tau = np.zeros(3)
        # get desired foot i pos (xi, yi, zi) in leg frame
        if LATERAL_MOTION:
          leg_xyz = np.array([0, sideSign[i] * foot_y + xs[i], zs[i]])  # motion selon y
        else:
          leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])  # motion selon x
        # leg_xyz = np.array([xs[i], sideSign[i] * foot_y + xs[i], zs[i]]) #motion selon x-y

        # Get current Jacobian and foot position in leg frame (see ComputeJacobianAndPosition() in quadruped.py)
        J, pos = env.robot.ComputeJacobianAndPosition(i)
        # call inverse kinematics to get corresponding joint angles (see ComputeInverseKinematics() in quadruped.py)
        leg_q = env.robot.ComputeInverseKinematics(i, leg_xyz)

        # add joint PD contribution
        if ADD_JOINT_PD:
          # Add joint PD contribution to tau for leg i (Equation 4)
          dq_d = np.zeros(3)
          tau += kp * (leg_q - q[i * 3:(i * 3 + 3)]) + kd * (dq_d - dq[i * 3:(i * 3 + 3)])

        # add Cartesian PD contribution
        if ADD_CARTESIAN_PD:
          # Get current foot velocity in leg frame (Equation 2)
          v = J @ dq[i * 3:(i * 3 + 3)]
          # Calculate torque contribution from Cartesian PD (Equation 5) [Make sure you are using matrix multiplications]
          v_d = np.zeros(3)
          tau += J.transpose() @ (kpCartesian @ (leg_xyz - pos) + kdCartesian @ (v_d - v))

        # Set tau for legi in action vector
        action[3 * i:3 * i + 3] = tau

        if VERBOSE:
          # save only for leg 0 (plot 2)
          if i == 0:
            # plot 2
            leg0_desired_foot_pos_x.append(leg_xyz[0])
            leg0_desired_foot_pos_y.append(leg_xyz[1])
            leg0_desired_foot_pos_z.append(leg_xyz[2])

            leg0_actual_foot_pos_x.append(pos[0])
            leg0_actual_foot_pos_y.append(pos[1])
            leg0_actual_foot_pos_z.append(pos[2])

            # plot 3
            leg0_desired_angle_hip.append(leg_q[0])
            leg0_desired_angle_thigh.append(leg_q[1])
            leg0_desired_angle_calf.append(leg_q[2])

            leg0_actual_angle_hip.append(q[0])
            leg0_actual_angle_thigh.append(q[1])
            leg0_actual_angle_calf.append(q[2])
      if VERBOSE:
        # get feet in contact to create a list
        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
        feetInContactBool_save_leg0.append(feetInContactBool[0])
        feetInContactBool_save_leg1.append(feetInContactBool[1])
        feetInContactBool_save_leg2.append(feetInContactBool[2])
        feetInContactBool_save_leg3.append(feetInContactBool[3])

      if cpg.gait == "TROT":
        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
        cmp_leg03.append((feetInContactBool[0] == feetInContactBool[3]))
        cmp_leg12.append((feetInContactBool[1] == feetInContactBool[2]))
        feetContactReward = (sum(cmp_leg03) + sum(cmp_leg12))
      elif cpg.gait == "PACE":
        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
        cmp_leg02.append((feetInContactBool[0] == feetInContactBool[2]))
        cmp_leg13.append((feetInContactBool[1] == feetInContactBool[3]))
        feetContactReward = (sum(cmp_leg02) + sum(cmp_leg13))
      elif cpg.gait == "BOUND":
        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
        cmp_leg01.append((feetInContactBool[0] == feetInContactBool[1]))
        cmp_leg23.append((feetInContactBool[2] == feetInContactBool[3]))
        feetContactReward = (sum(cmp_leg01) + sum(cmp_leg23))
      elif cpg.gait == "WALK":
        xyz = 0
      elif cpg.gait == "PRONK":
        numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
        cmp_leg0123.append((feetInContactBool[0] == feetInContactBool[1] == feetInContactBool[2] == feetInContactBool[3]))
      elif cpg.gait == "GALLOP":
        xyz = 0
      else:
        raise ValueError(cpg.gait + 'not implemented.')

      # send torques to robot and simulate TIME_STEP seconds
      env.step(action)

    # get final position of the base
    final_pos = env.robot.GetBasePosition()
    base_velocity = math.sqrt((final_pos[0] - initial_pos[0]) ** 2 + (final_pos[1] - initial_pos[1]) ** 2 + (
              final_pos[2] - initial_pos[2]) ** 2) / t[-1]
    print('Velocity of the base:', base_velocity, ' m/s')

    CoT = CoT / (MASS * GRAVITY * base_velocity)
    print('CoT divided by TEST_STEPS:', CoT / TEST_STEPS)

    if VERBOSE:
      # duty cycle and ratio calculation
      duty_cycle_list_leg0, duty_ratio_list_leg0 = duty_cycle_and_ratio(feetInContactBool_save_leg0)
      duty_cycle_list_leg1, duty_ratio_list_leg1 = duty_cycle_and_ratio(feetInContactBool_save_leg1)
      duty_cycle_list_leg2, duty_ratio_list_leg2 = duty_cycle_and_ratio(feetInContactBool_save_leg2)
      duty_cycle_list_leg3, duty_ratio_list_leg3 = duty_cycle_and_ratio(feetInContactBool_save_leg3)

    if VERBOSE:
      #####################################################
      # PLOTS
      #####################################################
      # What we should see:
      # 1) r_i (i = 1 à 4) should converge to sqrt(mu) (i.e limit cycle)
      # 2) TROT: theta_1 - theta4 = theta_2 - theta3 = 0   ------   theta_1 - theta3 = theta_2 - theta4 = np.pi (not sure about the sign?)
      # 3) change stance to swing as a fn of sin(theta_i)

      fig, axs = plt.subplots(2, 2)
      fig.suptitle('CPG states')
      axs[0, 0].plot(t, CPG_save_r1, label='r1')
      axs[0, 0].plot(t, CPG_save_r2, label='r2')
      axs[0, 0].plot(t, CPG_save_r3, label='r3')
      axs[0, 0].plot(t, CPG_save_r4, label='r4')
      axs[0, 0].plot(t, np.sqrt(cpg._mu) * np.ones(len(t)), label='mu')
      axs[0, 0].legend(loc="upper right")

      # axs[1,0].plot(t, CPG_save_theta1, label='theta1')
      axs[1, 0].plot(t, CPG_save_theta2, label='theta2')
      # axs[1,0].plot(t, CPG_save_theta3, label='theta3')
      axs[1, 0].plot(t, CPG_save_theta4, label='theta4')
      axs[1, 0].legend(loc="upper right")

      # axs[0,1].plot(t, [a - b for a, b in zip(CPG_save_theta1, CPG_save_theta2)], label='theta1 - theta2')
      # axs[0,1].plot(t, [a - b for a, b in zip(CPG_save_theta1, CPG_save_theta3)], label='theta1 - theta3')
      # axs[0,1].plot(t, [a - b for a, b in zip(CPG_save_theta1, CPG_save_theta4)], label='theta1 - theta4')
      # axs[0,1].plot(t, [a - b for a, b in zip(CPG_save_theta2, CPG_save_theta3)], label='theta2 - theta3')
      axs[0, 1].plot(t, [a - b for a, b in zip(CPG_save_theta2, CPG_save_theta4)], label='theta2 - theta4')
      axs[0, 1].plot(t, cpg.PHI[1, 3] * np.ones(len(t)), label='target PHI(2,4)')
      # axs[0,1].plot(t, [a - b for a, b in zip(CPG_save_theta3, CPG_save_theta4)], label='theta3 - theta4')
      axs[0, 1].legend(loc="upper right")

      # axs[1,1].plot(t, omega1_FSM_save, label='omega1')
      axs[1, 1].plot(t, omega2_FSM_save, label='omega2')
      # axs[1,1].plot(t, omega3_FSM_save, label='omega3')
      axs[1, 1].plot(t, omega4_FSM_save, label='omega4')
      axs[1, 1].legend(loc="upper right")

      plt.show()

      #####################################################
      # report plot 1 CPG states for each leg
      #####################################################

      fig1, axs1 = plt.subplots(2, 2)
      fig1.suptitle('CPG states for each leg')
      axs1[0, 0].plot(t, CPG_save_r2, label='r')
      axs1[0, 0].plot(t, CPG_save_theta2, label='theta')
      axs1[0, 0].plot(t, CPG_save_r2_dot, label='r_dot')
      axs1[0, 0].plot(t, CPG_save_theta2_dot, label='theta_dot')
      axs1[0, 0].legend(loc="upper right")
      axs1[0, 0].set_title("FL: leg(1)")

      axs1[1, 0].plot(t, CPG_save_r1, label='r')
      axs1[1, 0].plot(t, CPG_save_theta1, label='theta')
      axs1[1, 0].plot(t, CPG_save_r1_dot, label='r_dot')
      axs1[1, 0].plot(t, CPG_save_theta1_dot, label='theta_dot')
      axs1[1, 0].legend(loc="upper right")
      axs1[1, 0].set_title("FR: leg(0)")

      axs1[0, 1].plot(t, CPG_save_r4, label='r')
      axs1[0, 1].plot(t, CPG_save_theta4, label='theta')
      axs1[0, 1].plot(t, CPG_save_r4_dot, label='r_dot')
      axs1[0, 1].plot(t, CPG_save_theta4_dot, label='theta_dot')
      axs1[0, 1].legend(loc="upper right")
      axs1[0, 1].set_title("RL: leg(3)")

      axs1[1, 1].plot(t, CPG_save_r3, label='r')
      axs1[1, 1].plot(t, CPG_save_theta3, label='theta')
      axs1[1, 1].plot(t, CPG_save_r3_dot, label='r_dot')
      axs1[1, 1].plot(t, CPG_save_theta3_dot, label='theta_dot')
      axs1[1, 1].legend(loc="upper right")
      axs1[1, 1].set_title("RR: leg(2)")

      plt.show()

      #####################################################
      # report plot 2: Desired foot position VS actual foot position for leg 0
      #####################################################

      if ADD_CARTESIAN_PD and ADD_JOINT_PD:
        fig2, axs2 = plt.subplots(1, 3)
        fig2.suptitle('Desired foot position VS actual foot position for leg 0 with cartesian PD and joint PD')
        axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
        axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
        axs2[0].legend(loc="upper right")

        axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
        axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
        axs2[1].legend(loc="upper right")

        axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
        axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
        axs2[2].legend(loc="upper right")

      elif ADD_JOINT_PD:
        fig2, axs2 = plt.subplots(1, 3)
        fig2.suptitle('Desired foot position VS actual foot position for leg 0 with joint PD only')
        axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
        axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
        axs2[0].legend(loc="upper right")

        axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
        axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
        axs2[1].legend(loc="upper right")

        axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
        axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
        axs2[2].legend(loc="upper right")

      else:
        fig2, axs2 = plt.subplots(1, 3)
        fig2.suptitle('Desired foot position VS actual foot position for leg 0 with cartesian PD only')
        axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
        axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
        axs2[0].legend(loc="upper right")

        axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
        axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
        axs2[1].legend(loc="upper right")

        axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
        axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
        axs2[2].legend(loc="upper right")

      plt.show()

      #####################################################
      # report plot 3: Desired joint angles VS actual joint angles for leg 0
      #####################################################

      if ADD_CARTESIAN_PD and ADD_JOINT_PD:
        fig2, axs2 = plt.subplots(1, 3)
        fig2.suptitle('Desired joint angles VS actual joint angles for leg 0 with cartesian PD and joint PD')
        axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
        axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
        axs2[0].legend(loc="upper right")

        axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
        axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
        axs2[1].legend(loc="upper right")

        axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
        axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
        axs2[2].legend(loc="upper right")

      elif ADD_JOINT_PD:
        fig2, axs2 = plt.subplots(1, 3)
        fig2.suptitle('Desired joint angles VS actual joint angles for leg 0 with joint PD only')
        axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
        axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
        axs2[0].legend(loc="upper right")

        axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
        axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
        axs2[1].legend(loc="upper right")

        axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
        axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
        axs2[2].legend(loc="upper right")

      else:
        fig2, axs2 = plt.subplots(1, 3)
        fig2.suptitle('Desired joint angles VS actual joint angles for leg 0 with cartesian PD only')
        axs2[0].plot(t, leg0_desired_angle_hip, label='desired hip angle')
        axs2[0].plot(t, leg0_actual_angle_hip, label='actual hip angle')
        axs2[0].legend(loc="upper right")

        axs2[1].plot(t, leg0_desired_angle_thigh, label='desired thigh angle')
        axs2[1].plot(t, leg0_actual_angle_thigh, label='actual thigh angle')
        axs2[1].legend(loc="upper right")

        axs2[2].plot(t, leg0_desired_angle_calf, label='desired calf angle')
        axs2[2].plot(t, leg0_actual_angle_calf, label='actual calf angle')
        axs2[2].legend(loc="upper right")

      plt.show()

      #####################################################
      # CONTACT POINTS and duty cycles and ratios
      #####################################################

      # for TROT (0,3) (1,2) should be the same
      fig3, axs3 = plt.subplots(2, 2)
      fig3.suptitle('Contact forces and duty cycles/ratios')
      axs3[0, 0].plot(t, feetInContactBool_save_leg0, label='leg 0')
      axs3[0, 0].plot(t, feetInContactBool_save_leg3, label='leg 3')
      axs3[0, 0].legend(loc="upper right")
      axs3[0, 0].set_title("Contact forces (contact = 1 | no contact = 0)")

      axs3[0, 1].plot(t, feetInContactBool_save_leg1, label='leg 1')
      axs3[0, 1].plot(t, feetInContactBool_save_leg2, label='leg 2')
      axs3[0, 1].legend(loc="upper right")
      axs3[0, 1].set_title("Contact forces (contact = 1 | no contact = 0)")

      axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg0) + 1), duty_cycle_list_leg0, label='leg 0')
      axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg1) + 1), duty_cycle_list_leg1, label='leg 1')
      axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg2) + 1), duty_cycle_list_leg2, label='leg 2')
      axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg3) + 1), duty_cycle_list_leg3, label='leg 3')
      axs3[1, 0].legend(loc="upper right")
      axs3[1, 0].set_title("Duty cycle evolution")

      axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg0) + 1), duty_ratio_list_leg0, label='leg 0')
      axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg1) + 1), duty_ratio_list_leg1, label='leg 1')
      axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg2) + 1), duty_ratio_list_leg2, label='leg 2')
      axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg3) + 1), duty_ratio_list_leg3, label='leg 3')
      axs3[1, 1].legend(loc="upper right")
      axs3[1, 1].set_title("Duty ratio evolution")
      plt.show()

    env.reset()
