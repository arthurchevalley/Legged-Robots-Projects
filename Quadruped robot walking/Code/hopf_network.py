"""
CPG in polar coordinates based on: 
Pattern generators with sensory feedback for the control of quadruped
authors: L. Righetti, A. Ijspeert
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4543306

"""
import time
import numpy as np
import matplotlib
import math

from sys import platform

if platform == "darwin":  # mac
    import PyQt5

    matplotlib.use("Qt5Agg")
else:  # linux
    matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
from env.quadruped_gym_env import QuadrupedGymEnv

# Constants
TOTAL_SIM_TIME = 2
STANCE = 0
SWING = 1
MASS = 12
GRAVITY = 9.81
TARGET_ORIENTATION_YAW = np.pi/8

CONVERGENCE_LIMIT_CYCLE = 60
# Alpha                     TROT: 70  || PACE: 60  || BOUND: 50  ||  WALK: 70  ||  PRONK: 50  ||  GALLOP: 50
joint_PD_gains1 = 160
# P Gains Joint             TROT: 160  || PACE: 160  ||  BOUND: 150  || WALK: 160  ||  PRONK: 150  ||  GALLOP: 160
cartesian_PD_gains1 = 2500
# P Gains Cartesian         TROT: 2500  || PACE: 2500  ||  BOUND: 2500  ||  WALK: 2500  ||  PRONK: 2500  ||  GALLOP: 2500
joint_PD_gains2 = 3
# D Gains Joint             TROT: 2.5  || PACE: 2.5  ||  BOUND: 2  ||  WALK: 2.5  || PRONK: 2  ||  GALLOP: 2.5
cartesian_PD_gains2 = 40
# D Gains Cartesian         TROT: 44  || PACE: 44  ||  BOUND: 44  ||  WALK: 44  || PRONK: 40  ||  GALLOP: 40

kp = np.array([joint_PD_gains1 , joint_PD_gains1/2, joint_PD_gains1/2]) * 2
kd = np.array([joint_PD_gains2, joint_PD_gains2/4, joint_PD_gains2/4])

# Cartesian PD gains
kpCartesian = np.diag([cartesian_PD_gains1] * 3) * 2
kdCartesian = np.diag([cartesian_PD_gains2] * 3)

#define the controller used
ADD_CARTESIAN_PD = True
ADD_JOINT_PD = False

# define the number of simulation rounds (to compare with/without VMC_attitude)
NB_TESTS = 1

# define if plots should be shown at the end of the simulation
PLOTS = 1

# define motion direction (1 = forward, 0 = backward)
FORWARD_MOTION = 1

# define motion direction (LATERAL_MOTION: 0 = moving along x, 1 = moving along y or DIAGONAL_MOTION to move along x-y)
LATERAL_MOTION = 0
DIAGONAL_MOTION = 0

# enable VMC roll and pitch rejection of the base
VMC_attitude = 0

# enable VMC orientation tracking of the base
VMC_orientation = 0


class HopfNetwork():
    """ CPG network based on hopf polar equations mapped to foot positions in Cartesian space.

  Foot Order is FR, FL, RR, RL
  (Front Right, Front Left, Rear Right, Rear Left)
  """

    def __init__(self,
                 mu=1 ** 2,  # converge to sqrt(mu)
                 omega_swing = 48 * np.pi,
                 # Swing frequency                                     TROT 16*np.pi || PACE 16*np.pi  || BOUND 16*np.pi || WALK 18*np.pi  || PRONK 8*np.pi  || GALLOP 24*np.pi
                 omega_stance = 12 * np.pi,
                 # Stance Frequency                                    TROT 8*np.pi || PACE 4*np.pi  || BOUND 4*np.pi || WALK 6*np.pi  || PRONK 17*np.pi  || GALLOP 8*np.pi
                 gait="PACE",
                 # change depending on desired gait
                 coupling_strength=1.2,
                 # coefficient to multiply coupling matrix              TROT 1.2 || PACE 1.2 || BOUND 1 || WALK  1 || PRONK 1.1  || GALLOP  1.2
                 couple=True,
                 # should couple boolean
                 time_step=0.001,
                 # time step
                 ground_clearance=0.04,
                 #foot swing height gc                                   TROT 0.05 || PACE 0.04 || BOUND 0.04 || WALK 0.06 || PRONK 0.04 || GALLOP  0.05
                 ground_penetration=0.01,
                 # foot stance penetration into ground gp 0.01           TROT 0.01 || PACE 0.01 || BOUND 0.01 || WALK 0.01  || PRONK 0.01  || GALLOP 0.01
                 robot_height=0.26,
                 # robot height in nominal case (standing)  0.25                      TROT 0.27 || PACE 0.26 || BOUND 0.27 || WALK 0.27  || PRONK 0.2  || GALLOP 0.24
                 des_step_len=0.04,
                 # desired step length 0.04                              TROT 0.04 || PACE 0.04 || BOUND 0.04 || WALK 0.04 || PRONK 0.035  || GALLOP 0.03
                 ):

        ###############
        # initialize CPG data structures: amplitude is row 0, and phase is row 1
        self.X = np.zeros((2, 4))

        # save parameters
        self._mu = mu
        self._omega_swing = omega_swing
        self._omega_stance = omega_stance
        self._couple = couple
        self._coupling_strength = coupling_strength
        self._dt = time_step
        self._set_gait(gait)

        # me to plot the change between stance and swing phase
        self.omega_FSM = np.zeros(4)

        # me to plot r_dot and theta_dot for each leg
        self.X_dot = np.zeros((2, 4))

        # set oscillator initial conditions
        self.X[0, :] = np.random.rand(4) * .1
        self.X[1, :] = self.PHI[0, :]

        # save body and foot shaping
        self._ground_clearance = ground_clearance
        self._ground_penetration = ground_penetration
        self._robot_height = robot_height
        self._des_step_len = des_step_len

    def _set_gait(self, gait):
        """ For coupling oscillators in phase space.
    """
        self.PHI_trot = np.array(
            [[0, np.pi, np.pi, 0], [-np.pi, 0, 0, -np.pi], [-np.pi, 0, 0, -np.pi], [0, np.pi, np.pi, 0]])
        self.PHI_bound = np.array(
            [[0, 0, -np.pi, -np.pi], [0, 0, -np.pi, -np.pi], [np.pi, np.pi, 0, 0], [np.pi, np.pi, 0, 0]])
        self.PHI_pace = np.array(
            [[0, np.pi, 0, np.pi], [-np.pi, 0, -np.pi, 0], [0, np.pi, 0, np.pi], [-np.pi, 0, -np.pi, 0]])
        self.PHI_walk = np.array(
            [[0, np.pi, 3 * np.pi / 2, np.pi / 2], [-np.pi, 0, np.pi / 2, -np.pi / 2],
             [-3 * np.pi / 2, -np.pi / 2, 0, -np.pi], [-np.pi / 2, np.pi / 2, np.pi, 0]])
        self.PHI_pronk = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.PHI_gallop = np.array([[0, -np.pi / 5, -6 * np.pi / 5, -np.pi], [np.pi / 5, 0, -np.pi, -4 * np.pi / 5],
                                    [6 * np.pi / 5, np.pi, 0, np.pi / 5], [np.pi, 4 * np.pi / 5, -np.pi / 5, 0]])
        if gait == "TROT":
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
            raise ValueError(gait + 'not implemented.')

    def update(self):
        """ Update oscillator states. """

        # update parameters, integrate
        self._integrate_hopf_equations()

        x = np.zeros(len(self.X[0, :]))
        z = np.zeros(len(self.X[0, :]))
        # map CPG variables to Cartesian foot xz positions (Equations 8, 9)
        for i in range(len(self.X[0, :])):

            if FORWARD_MOTION:
                x[i] = - self._des_step_len * self.X[0, i] * np.cos(self.X[1, i])  # forward motion
            else:
                x[i] = self._des_step_len * self.X[0, i] * np.cos(self.X[1, i])  # backward motion

            if np.sin(self.X[1, i]) > 0:
                z[i] = - self._robot_height + self._ground_clearance * np.sin(self.X[1, i])
            else:
                z[i] = - self._robot_height + self._ground_penetration * np.sin(self.X[1, i])

        return x, z

    def _integrate_hopf_equations(self):
        """ Hopf polar equations and integration. Use equations 6 and 7. """
        # bookkeeping - save copies of current CPG states
        nb_oscillators = len(self.X[0, :])
        X = self.X.copy()
        X_dot = np.zeros((2, 4))
        alpha = CONVERGENCE_LIMIT_CYCLE

        # loop through each leg's oscillator
        for i in range(nb_oscillators):
            # get r_i, theta_i from X
            r, theta = X[0, i], X[1, i]
            # compute r_dot (Equation 6)
            r_dot = alpha * (self._mu - r ** 2) * r
            # determine whether oscillator i is in swing or stance phase to set natural frequency omega_swing or omega_stance (see Section 3)
            if np.sin(theta % (2 * np.pi)) > 0:
                theta_dot = self._omega_swing
            else:
                theta_dot = self._omega_stance

            # to plot
            self.omega_FSM[i] = theta_dot

            # loop through other oscillators to add coupling (Equation 7)
            if self._couple:
                for j in range(nb_oscillators):
                    # ATTENTION TO CHECK
                    theta_dot += X[0, j] * self._coupling_strength * np.sin(
                        X[1, j] - X[1, i] - self.PHI[i, j])

            # set X_dot[:,i]
            self.X_dot[:, i] = [r_dot, theta_dot]

        # integrate
        self.X += self.X_dot * self._dt

        # mod phase variables to keep between 0 and 2pi
        self.X[1, :] = self.X[1, :] % (2 * np.pi)


def Virtual_model_control_attitude_base(env):
    """ Computes the torque tau (12x1) to be applied to the leg joints to maintain the base attitude parallel to the ground"""

    k_spring = 250 #250
    R = env.robot.GetBaseOrientationMatrix()
    P = R @ np.array([[1, 1, -1, -1], [-1, 1, -1, 1], [0, 0, 0, 0]])

    Fz_all_legs = k_spring * np.transpose([0, 0, 1] @ P)

    F_hip_leg = np.array([0, 0, Fz_all_legs[0], 0, 0, Fz_all_legs[1], 0, 0, Fz_all_legs[2], 0, 0, Fz_all_legs[3]])
    F_feet_leg = - F_hip_leg

    tau = np.zeros(12)
    for i in range(4):
        J, pos = env.robot.ComputeJacobianAndPosition(i)
        # tau.append(np.transpose(J)@F_feet_leg[i*3:(i*3+3)])
        tau[3 * i:3 * i + 3] = np.transpose(J) @ F_feet_leg[i * 3:(i * 3 + 3)]
    return tau

def Virtual_model_control_orientation_base(env):
    """ Computes the torque tau (12x1) to be applied to the leg joints for the base orientation to match the target orientation """

    k_orien = 80

    roll_pitch_yaw = env.robot.GetBaseOrientationRollPitchYaw()

    delta = TARGET_ORIENTATION_YAW - roll_pitch_yaw[2]

    tau = np.zeros(12)
    F_feet_leg = np.zeros(12)

    F_hip_leg = k_orien * np.array([np.cos(delta), np.sin(delta), 0, np.cos(delta), np.sin(delta), 0, -np.cos(delta), -np.sin(delta), 0, -np.cos(delta), -np.sin(delta), 0])
    F_feet_leg = - F_hip_leg
    for i in range(4):
        J, pos = env.robot.ComputeJacobianAndPosition(i)
        tau[3 * i:3 * i + 3] = np.transpose(J) @ F_feet_leg[i * 3:(i * 3 + 3)]

    return F_feet_leg, tau

def duty_cycle_and_ratio(feetInContactBool):  # feetInContactBool = list of 1 and 0 for a single leg
    """ Computes the average duty cycle (T_stance + T_swing) and average the duty ratio D = T_stance / T_swing"""

    T_stance_list = []
    T_swing_list = []
    duty_cycle_list = []
    duty_ratio_list = []

    duty_cycle_mean = 0
    duty_ratio_mean = 0
    i = 0

    while i < len(feetInContactBool):

        if feetInContactBool[i] == STANCE:
            T_stance = 0
            while i < len(feetInContactBool) and feetInContactBool[i] != SWING:
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

    count1= 0
    for i in range(len(duty_cycle_list)):
        if i > 0 and i < (len(duty_cycle_list) - 1): #to avoid init and final step
            duty_cycle_mean = duty_cycle_mean + duty_cycle_list[i]
            count1 = count1 + 1

    duty_cycle_mean = duty_cycle_mean/count1

    count1 = 0
    for i in range(len(duty_ratio_list)):
        if i > 0 and i < (len(duty_ratio_list) - 1):  # to avoid init and final step
            duty_ratio_mean = duty_ratio_mean + duty_ratio_list[i]
            count1 = count1 + 1

    duty_ratio_mean = duty_ratio_mean / count1

    return duty_cycle_mean, duty_ratio_mean


if __name__ == "__main__":

    TIME_STEP = 0.001
    foot_y = 0.0838  # this is the hip length
    sideSign = np.array([-1, 1, -1, 1])  # get correct hip sign (body right is negative)

    VMC_roll = []
    VMC_pitch = []

    env = QuadrupedGymEnv(render=True,  # visualize
                            on_rack=False,  # useful for debugging!
                            isRLGymInterface=False,  # not using RL
                            time_step=TIME_STEP,
                            action_repeat=1,
                            motor_control_mode="TORQUE",
                            add_noise=False,  # start in ideal conditions
                            test_env=False #record_video=True
                            )

    # initialize Hopf Network, supply gait
    cpg = HopfNetwork(time_step=TIME_STEP)

    TEST_STEPS = int(TOTAL_SIM_TIME / (TIME_STEP))
    t = np.arange(TEST_STEPS) * TIME_STEP

    for r in range(NB_TESTS):
      if r == 0:
        VMC_attitude = 0
        print('Without VMC attitude')
      else:
        env.reset()
        VMC_attitude = 1
        print('With VMC attitude')
      # [TODO] initialize data structures to save CPG and robot states
      # Save CPG
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
      duty_cycle_mean_leg0 = 0
      duty_cycle_mean_leg1 = 0
      duty_cycle_mean_leg2 = 0
      duty_cycle_mean_leg3 = 0

      duty_ratio_mean_leg0 = 0
      duty_ratio_mean_leg1 = 0
      duty_ratio_mean_leg2 = 0
      duty_ratio_mean_leg3 = 0

      # to compare with and without VMC
      roll_base = []
      pitch_base = []
      yaw_base = []
      VMC_orientation_applied_or_not = []

      #remove
      Fx_H1_list = []
      Fy_H1_list = []
      Fx_H2_list = []
      Fy_H2_list = []
      Fx_H3_list = []
      Fy_H3_list = []
      Fx_H4_list = []
      Fy_H4_list = []

      # record hip angle for leg0 (to debug VMC ORIENTATION)
      action_PD_list_leg0 = []
      action_orientation_list_leg0 = []
      torque_applied_list_sim_leg0 = []
      action_applied_list_leg0 = []

      # record hip angle for leg1 (to debug VMC ORIENTATION)
      action_PD_list_leg1 = []
      action_orientation_list_leg1 = []
      torque_applied_list_sim_leg1 = []
      action_applied_list_leg1 = []

      # record hip angle for leg2 (to debug VMC ORIENTATION)
      action_PD_list_leg2 = []
      action_orientation_list_leg2 = []
      torque_applied_list_sim_leg2 = []
      action_applied_list_leg2 = []

      # record hip angle for leg3 (to debug VMC ORIENTATION)
      action_PD_list_leg3 = []
      action_orientation_list_leg3 = []
      torque_applied_list_sim_leg3 = []
      action_applied_list_leg3 = []

      ############## Sample Gains ##############
      # joint PD gains
      #  put back here


      # get initial position of the base
      initial_pos = env.robot.GetBasePosition()

      CoT = 0
      for j in range(TEST_STEPS):

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
              elif DIAGONAL_MOTION:
                  leg_xyz = np.array([xs[i], sideSign[i] * foot_y + xs[i], zs[i]])  # motion selon y
              else:#forward
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

          # get feet in contact to create a list
          numValidContacts, numInvalidContacts, feetNormalForces, feetInContactBool = env.robot.GetContactInfo()
          feetInContactBool_save_leg0.append(feetInContactBool[0])
          feetInContactBool_save_leg1.append(feetInContactBool[1])
          feetInContactBool_save_leg2.append(feetInContactBool[2])
          feetInContactBool_save_leg3.append(feetInContactBool[3])

          action_PD_list_leg0.append(action[0])
          action_PD_list_leg1.append(action[3])
          action_PD_list_leg2.append(action[6])
          action_PD_list_leg3.append(action[9])
          # get feet in contact to create a list
          nb_feet_touching_gnd = sum(feetInContactBool)
          # add attitude control
          if VMC_attitude and nb_feet_touching_gnd > 1:
              #print('With VMC attitude')
              action_attitude = np.zeros(12)
              tau_attitude = Virtual_model_control_attitude_base(env)

              # apply force only for feet touching the ground and only if the number of feet touching the ground >1
              for j in range(4):
                  if feetInContactBool[j]:
                      action_attitude[3 * j:3 * j + 3] = tau_attitude[3 * j:3 * j + 3]

              action += action_attitude

          # add orientation control
          if VMC_orientation and nb_feet_touching_gnd > 1:
            VMC_orientation_applied_or_not.append(1)
            #print('With VMC orientation')
            action_orientation = np.zeros(12)
            F, tau_orientation = Virtual_model_control_orientation_base(env)

            Fx_H1_list.append(F[0])
            Fy_H1_list.append(F[1])

            Fx_H2_list.append(F[3])
            Fy_H2_list.append(F[4])

            Fx_H3_list.append(F[6])
            Fy_H3_list.append(F[7])

            Fx_H4_list.append(F[9])
            Fy_H4_list.append(F[10])


            # apply force only for feet touching the ground
            for j in range(4):
                if feetInContactBool[j]:
                    action_orientation[3 * j:3 * j + 3] = tau_orientation[3 * j:3 * j + 3]

            action_orientation_list_leg0.append(action_orientation[0])
            action_orientation_list_leg1.append(action_orientation[3])
            action_orientation_list_leg2.append(action_orientation[6])
            action_orientation_list_leg3.append(action_orientation[9])

            action += action_orientation

          else:
              VMC_orientation_applied_or_not.append(0)
              # only joint 0
              action_orientation_list_leg0.append(0)
              action_orientation_list_leg1.append(0)
              action_orientation_list_leg2.append(0)
              action_orientation_list_leg3.append(0)

          # save roll and pitch and yaw for plots (VMC)
          roll_pitch_yaw = env.robot.GetBaseOrientationRollPitchYaw()
          roll_base.append(roll_pitch_yaw[0])
          pitch_base.append(roll_pitch_yaw[1])
          yaw_base.append(roll_pitch_yaw[2])

          action_applied_list_leg0.append(action[0])
          action_applied_list_leg1.append(action[3])
          action_applied_list_leg2.append(action[6])
          action_applied_list_leg3.append(action[9])
          # send torques to robot and simulate TIME_STEP seconds
          env.step(action)


          #get torque applied
          #if VMC_orientation and nb_feet_touching_gnd > 1:
          torques = env.robot.GetMotorTorques()
          torque_applied_list_sim_leg0.append(torques[0])
          torque_applied_list_sim_leg1.append(torques[3])
          torque_applied_list_sim_leg2.append(torques[6])
          torque_applied_list_sim_leg3.append(torques[9])

      # get final position of the base
      final_pos = env.robot.GetBasePosition()

      base_velocity = math.sqrt((final_pos[0] - initial_pos[0]) ** 2 + (final_pos[1] - initial_pos[1]) ** 2 + (
                  final_pos[2] - initial_pos[2]) ** 2) / t[-1]

      print('Velocity of the base:', base_velocity, ' m/s')

      # j'ai mis divisé par TEST_STEPS mais je suis pas sure
      CoT = CoT / (MASS * GRAVITY * base_velocity)
      print('CoT:', CoT)
      print('CoT divided by TEST_STEPS:', CoT / TEST_STEPS)

      # duty cycle and ratio calculation
      duty_cycle_mean_leg0, duty_ratio_mean_leg0 = duty_cycle_and_ratio(feetInContactBool_save_leg0)
      duty_cycle_mean_leg1, duty_ratio_mean_leg1 = duty_cycle_and_ratio(feetInContactBool_save_leg1)
      duty_cycle_mean_leg2, duty_ratio_mean_leg2 = duty_cycle_and_ratio(feetInContactBool_save_leg2)
      duty_cycle_mean_leg3, duty_ratio_mean_leg3 = duty_cycle_and_ratio(feetInContactBool_save_leg3)

      duty_cycle_mean = (duty_cycle_mean_leg0 + duty_cycle_mean_leg1 + duty_cycle_mean_leg2 + duty_cycle_mean_leg3)/4
      duty_ratio_mean = (duty_ratio_mean_leg0 + duty_ratio_mean_leg1 + duty_ratio_mean_leg2 + duty_ratio_mean_leg3)/4
      print('Duty cycle mean: ', duty_cycle_mean, ' [s]')
      print('Duty ratio mean: ', duty_ratio_mean)

      VMC_roll.append(roll_base)
      VMC_pitch.append(pitch_base)



    if PLOTS:
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
      axs1[0, 0].set_xlabel('Time [s]')
      axs1[0, 0].set_ylabel('CPG states for LF(L1)')
      axs1[0, 0].set_title("LF(L1)")

      axs1[1, 0].plot(t, CPG_save_r1, label='r')
      axs1[1, 0].plot(t, CPG_save_theta1, label='theta')
      axs1[1, 0].plot(t, CPG_save_r1_dot, label='r_dot')
      axs1[1, 0].plot(t, CPG_save_theta1_dot, label='theta_dot')
      axs1[1, 0].legend(loc="upper right")
      axs1[1, 0].set_xlabel('Time [s]')
      axs1[1, 0].set_ylabel('CPG states for RF(L0)')
      axs1[1, 0].set_title("RF(L0)")

      axs1[0, 1].plot(t, CPG_save_r4, label='r')
      axs1[0, 1].plot(t, CPG_save_theta4, label='theta')
      axs1[0, 1].plot(t, CPG_save_r4_dot, label='r_dot')
      axs1[0, 1].plot(t, CPG_save_theta4_dot, label='theta_dot')
      axs1[0, 1].legend(loc="upper right")
      axs1[0, 1].set_xlabel('Time [s]')
      axs1[0, 1].set_ylabel('CPG states for LH(L3)')
      axs1[0, 1].set_title("LH(L3)")

      axs1[1, 1].plot(t, CPG_save_r3, label='r')
      axs1[1, 1].plot(t, CPG_save_theta3, label='theta')
      axs1[1, 1].plot(t, CPG_save_r3_dot, label='r_dot')
      axs1[1, 1].plot(t, CPG_save_theta3_dot, label='theta_dot')
      axs1[1, 1].legend(loc="upper right")
      axs1[1, 1].set_xlabel('Time [s]')
      axs1[1, 1].set_ylabel('CPG states for RH(L2)')
      axs1[1, 1].set_title("RH(L2)")

      plt.show()

      #####################################################
      # report plot 2: Desired foot position VS actual foot position for leg 0
      #####################################################

      if ADD_CARTESIAN_PD and ADD_JOINT_PD:
          fig2, axs2 = plt.subplots(1, 3)
          fig2.suptitle('Desired foot position VS actual foot position for leg 0 with cartesian PD and joint PD')
          axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
          axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
          axs2[0].set_xlabel('Time [s]')
          axs2[0].set_ylabel('Foot x position [m]')
          axs2[0].legend(loc="upper right")

          axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
          axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
          axs2[1].set_xlabel('Time [s]')
          axs2[1].set_ylabel('Foot y position [m]')
          axs2[1].legend(loc="upper right")

          axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
          axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
          axs2[2].set_xlabel('Time [s]')
          axs2[2].set_ylabel('Foot z position [m]')
          axs2[2].legend(loc="upper right")

      elif ADD_JOINT_PD:
          fig2, axs2 = plt.subplots(1, 3)
          fig2.suptitle('Desired foot position VS actual foot position for leg 0 with joint PD only')
          axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
          axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
          axs2[0].set_xlabel('Time [s]')
          axs2[0].set_ylabel('Foot x position [m]')
          axs2[0].legend(loc="upper right")

          axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
          axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
          axs2[1].set_xlabel('Time [s]')
          axs2[1].set_ylabel('Foot y position [m]')
          axs2[1].legend(loc="upper right")

          axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
          axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
          axs2[2].set_xlabel('Time [s]')
          axs2[2].set_ylabel('Foot z position [m]')
          axs2[2].legend(loc="upper right")

      else:
          fig2, axs2 = plt.subplots(1, 3)
          fig2.suptitle('Desired foot position VS actual foot position for leg 0 with cartesian PD only')
          axs2[0].plot(t, leg0_desired_foot_pos_x, label='desired x')
          axs2[0].plot(t, leg0_actual_foot_pos_x, label='actual x')
          axs2[0].set_xlabel('Time [s]')
          axs2[0].set_ylabel('Foot x position [m]')
          axs2[0].legend(loc="upper right")

          axs2[1].plot(t, leg0_desired_foot_pos_y, label='desired y')
          axs2[1].plot(t, leg0_actual_foot_pos_y, label='actual y')
          axs2[1].set_xlabel('Time [s]')
          axs2[1].set_ylabel('Foot y position [m]')
          axs2[1].legend(loc="upper right")

          axs2[2].plot(t, leg0_desired_foot_pos_z, label='desired z')
          axs2[2].plot(t, leg0_actual_foot_pos_z, label='actual z')
          axs2[2].set_xlabel('Time [s]')
          axs2[2].set_ylabel('Foot z position [m]')
          axs2[2].legend(loc="upper right")

      plt.show()

      #####################################################
      # report plot 3: Desired joint angles VS actual joint angles for leg 0
      #####################################################

      if ADD_CARTESIAN_PD and ADD_JOINT_PD:
          fig2, axs2 = plt.subplots(1, 3)
          fig2.suptitle('Desired joint angles VS actual joint angles for leg 0 with cartesian PD and joint PD')
          axs2[0].plot(t, leg0_desired_angle_hip, label='desired hip angle')
          axs2[0].plot(t, leg0_actual_angle_hip, label='actual hip angle')
          axs2[0].set_xlabel('Time [s]')
          axs2[0].set_ylabel('Hip angle [rad]')
          axs2[0].legend(loc="upper right")

          axs2[1].plot(t, leg0_desired_angle_thigh, label='desired thigh angle')
          axs2[1].plot(t, leg0_actual_angle_thigh, label='actual thigh angle')
          axs2[1].set_xlabel('Time [s]')
          axs2[1].set_ylabel('Thigh angle [rad]')
          axs2[1].legend(loc="upper right")

          axs2[2].plot(t, leg0_desired_angle_calf, label='desired calf angle')
          axs2[2].plot(t, leg0_actual_angle_calf, label='actual calf angle')
          axs2[2].set_xlabel('Time [s]')
          axs2[2].set_ylabel('Calf angle [rad]')
          axs2[2].legend(loc="upper right")

      elif ADD_JOINT_PD:
          fig2, axs2 = plt.subplots(1, 3)
          fig2.suptitle('Desired joint angles VS actual joint angles for leg 0 with joint PD only')
          axs2[0].plot(t, leg0_desired_angle_hip, label='desired hip angle')
          axs2[0].plot(t, leg0_actual_angle_hip, label='actual hip angle')
          axs2[0].set_xlabel('Time [s]')
          axs2[0].set_ylabel('Hip angle [rad]')
          axs2[0].legend(loc="upper right")

          axs2[1].plot(t, leg0_desired_angle_thigh, label='desired thigh angle')
          axs2[1].plot(t, leg0_actual_angle_thigh, label='actual thigh angle')
          axs2[1].set_xlabel('Time [s]')
          axs2[1].set_ylabel('Thigh angle [rad]')
          axs2[1].legend(loc="upper right")

          axs2[2].plot(t, leg0_desired_angle_calf, label='desired calf angle')
          axs2[2].plot(t, leg0_actual_angle_calf, label='actual calf angle')
          axs2[2].set_xlabel('Time [s]')
          axs2[2].set_ylabel('Calf angle [rad]')
          axs2[2].legend(loc="upper right")

      else:
          fig2, axs2 = plt.subplots(1, 3)
          fig2.suptitle('Desired joint angles VS actual joint angles for leg 0 with cartesian PD only')
          axs2[0].plot(t, leg0_desired_angle_hip, label='desired hip angle')
          axs2[0].plot(t, leg0_actual_angle_hip, label='actual hip angle')
          axs2[0].set_xlabel('Time [s]')
          axs2[0].set_ylabel('Hip angle [rad]')
          axs2[0].legend(loc="upper right")

          axs2[1].plot(t, leg0_desired_angle_thigh, label='desired thigh angle')
          axs2[1].plot(t, leg0_actual_angle_thigh, label='actual thigh angle')
          axs2[1].set_xlabel('Time [s]')
          axs2[1].set_ylabel('Thigh angle [rad]')
          axs2[1].legend(loc="upper right")

          axs2[2].plot(t, leg0_desired_angle_calf, label='desired calf angle')
          axs2[2].plot(t, leg0_actual_angle_calf, label='actual calf angle')
          axs2[2].set_xlabel('Time [s]')
          axs2[2].set_ylabel('Calf angle [rad]')
          axs2[2].legend(loc="upper right")

      plt.show()

      #####################################################
      # CONTACT POINTS and duty cycles and ratios
      #####################################################

      # for TROT (0,3) (1,2) should be the same
      fig3, axs3 = plt.subplots(2, 1)
      fig3.suptitle('Contact forces and duty cycles/ratios')
      axs3[0].plot(t, feetInContactBool_save_leg0, label='L0')
      axs3[0].plot(t, feetInContactBool_save_leg3, label='L3')
      axs3[0].legend(loc="upper right")
      axs3[0].set_xlabel('Time [s]')
      axs3[0].set_ylabel('Feet contact booleans')
      axs3[0].set_title("Contact forces (contact = 1 | no contact = 0)")

      axs3[1].plot(t, feetInContactBool_save_leg1, label='L1')
      axs3[1].plot(t, feetInContactBool_save_leg2, label='L2')
      axs3[1].legend(loc="upper right")
      axs3[1].set_xlabel('Time [s]')
      axs3[1].set_ylabel('Feet contact booleans')
      axs3[1].set_title("Contact forces (contact = 1 | no contact = 0)")
      plt.show()
      #axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg0) + 1), duty_cycle_list_leg0, label='L0')
      #axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg1) + 1), duty_cycle_list_leg1, label='L1')
      #axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg2) + 1), duty_cycle_list_leg2, label='L2')
      #axs3[1, 0].plot(np.arange(1, len(duty_cycle_list_leg3) + 1), duty_cycle_list_leg3, label='L3')
      #axs3[1, 0].legend(loc="upper right")
      #axs3[1, 0].set_xlabel('Time [s]')
      #axs3[1, 0].set_ylabel('Duty cycle in [s]')
      #axs3[1, 0].set_title("Duty cycle evolution")

      #axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg0) + 1), duty_ratio_list_leg0, label='L0')
      #axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg1) + 1), duty_ratio_list_leg1, label='L1')
      #axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg2) + 1), duty_ratio_list_leg2, label='L2')
      #axs3[1, 1].plot(np.arange(1, len(duty_ratio_list_leg3) + 1), duty_ratio_list_leg3, label='L3')
      #axs3[1, 1].legend(loc="upper right")
      #axs3[1, 1].set_xlabel('Time [s]')
      #axs3[1, 1].set_ylabel('Duty ratio')
      #axs3[1, 1].set_title("Duty ratio evolution")
      #

      if VMC_orientation:
          fig4, axs4 = plt.subplots(2, 1) #4,2
          fig4.suptitle('VMC orientation')
          axs4[0].plot(t, VMC_orientation_applied_or_not)
          axs4[0].legend(loc="upper right")
          axs4[0].set_title("VMC applied (nb contact > 1) (1 = yes, 0 = no)")

          axs4[1].plot(t, TARGET_ORIENTATION_YAW*np.ones(len(t)), label='yaw target')
          axs4[1].plot(t, yaw_base, label='yaw real')
          axs4[1].legend(loc="upper right")
          axs4[1].set_title("Yaw evolution")
          plt.show()

      # plots VMC attitude as a fn of time
      if VMC_attitude:
        figV, axsV = plt.subplots(1, 3)
        figV.suptitle('Roll evolution: with VMC VS without VMC')
        axsV[0].plot(t, VMC_roll[0], label='Roll without VMC')
        axsV[0].plot(t, VMC_roll[1], label='Roll with VMC')
        axsV[0].legend(loc="upper right")

        axsV[1].plot(t, VMC_pitch[0], label='Pitch without VMC')
        axsV[1].plot(t, VMC_pitch[1], label='Pitch with VMC')
        axsV[1].legend(loc="upper right")

        # other plots VMC attitude (roll as a fn of pitch)
        axsV[2].plot(VMC_roll[0], VMC_pitch[0], label='Pitch VS roll without VMC')
        axsV[2].plot(VMC_roll[1], VMC_pitch[1], label='Pitch VS roll with VMC')
        axsV[2].legend(loc="upper right")
        plt.show()