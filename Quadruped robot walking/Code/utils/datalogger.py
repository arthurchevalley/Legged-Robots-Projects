import numpy as np
import matplotlib.pyplot as plt
import math


class DATALOGGER:

    def __init__(self):
        self.logs = {
            "rewards": [],
            "base_pos": [],
            "base_orientation": [],
            "base_linear_velocity": [],
            "base_angular_velocity": [],
            "motor_angle": [],
            "motor_velocity": [],
            "motor_torque": [],
            "feet_position": [],
            "feet_velocity": [],
            "feet_contact_bool": [],
            "time": []
        }

    def log(self, rewards, base_pos, base_orientation, base_linear_velocity, base_angular_velocity, motor_angle,
            motor_velocity, motor_torque, feet_position, feet_velocity, feet_contact_bool, time):
        self.logs['rewards'].append(rewards)
        self.logs['base_pos'].append(base_pos)
        self.logs['base_orientation'].append(base_orientation)
        self.logs['base_linear_velocity'].append(base_linear_velocity)
        self.logs['base_angular_velocity'].append(base_angular_velocity)
        self.logs['motor_angle'].append(motor_angle)
        self.logs['motor_velocity'].append(motor_velocity)
        self.logs['motor_torque'].append(motor_torque)
        self.logs['feet_position'].append(feet_position)
        self.logs['feet_velocity'].append(feet_velocity)
        self.logs['feet_contact_bool'].append(feet_contact_bool)
        self.logs['time'].append(time)

    def plot(self, start=100, stop=500):
        time = np.array(self.logs['time'])
        base_pos = np.array(self.logs['base_pos'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], base_pos[start:stop, 2])
        else:
            plt.plot(time, base_pos[:, 2])
        plt.title("base z")
        plt.show()

        time = np.array(self.logs['time'])
        base_pos = np.array(self.logs['base_pos'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], base_pos[start:stop, 0])
            plt.plot(time[start:stop], base_pos[start:stop, 1])
        else:
            plt.plot(time, base_pos[:, 0])
            plt.plot(time, base_pos[:, 1])
        plt.title("base x,y")
        plt.legend(['x', 'y'])
        plt.show()

        # plot the reward
        time = np.array(self.logs['time'])
        rewards = np.array(self.logs['rewards'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], rewards[start:stop, 0])
        else:
            plt.plot(time, rewards[:, 0])
        plt.title("Reward")
        plt.show()

        base_orientation = np.array(self.logs['base_orientation'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], base_orientation[start:stop, :])
        else:
            plt.plot(time, base_orientation)
        plt.title("Base orientation")
        plt.legend(['roll', 'pitch', 'yaw'])
        plt.show()

        base_angular_velocity = np.array(self.logs['base_angular_velocity'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], base_angular_velocity[start:stop, :])
        else:
            plt.plot(time, base_angular_velocity)
        plt.title("Base angular velocity")
        plt.legend(['roll', 'pitch', 'yaw'])
        plt.show()

        feet_contact_bool = np.array(self.logs['feet_contact_bool'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], feet_contact_bool[start:stop, :])
        else:
            plt.plot(time, feet_contact_bool)
        plt.title("Contact feet")
        plt.legend(['1', '2', '3', '4'])
        plt.show()

        base_linear_velocity = np.array(self.logs['base_linear_velocity'])
        if start is not None and stop is not None:
            plt.plot(time[start:stop], base_linear_velocity[start:stop, :])
        else:
            plt.plot(time, base_linear_velocity)
        plt.title("Base Linear Velocity")
        plt.legend(['x', 'y', 'z'])
        plt.show()

    def CoT_simu(self, nb_step_evaluation):
        CoT = 0
        MASS = 12
        GRAVITY = 9.81
        TIME_STEP = 0.001

        time = np.array(self.logs['time'])
        base_pos = np.array(self.logs['base_pos'])
        motor_torque = np.array(self.logs['motor_torque'])
        motor_velocity = np.array(self.logs['motor_velocity'])

        base_velocity = math.sqrt((base_pos[-1][0] - base_pos[0][0]) ** 2 + (base_pos[-1][1] - base_pos[0][1]) ** 2) / time[-1]
        print('base_velocity : ', base_velocity)

        for i in range(nb_step_evaluation):
            for j in range(len(motor_torque[0])):
                CoT += abs(motor_velocity[i][j] * motor_torque[i][j])

        CoT = CoT / (MASS * GRAVITY * base_velocity)
        print('CoT:', CoT)
        print('CoT divided by TEST_STEPS:', CoT / nb_step_evaluation)

    def duty_cycle_and_ratio(self, feetInContactBool):  # feetInContactBool = list of 1 and 0 for a single leg
        """ Computes the average duty cycle (T_stance + T_swing) and average the duty ratio D = T_stance / T_swing"""
        STANCE = 0
        SWING = 1
        T_stance_list = []
        T_swing_list = []
        duty_cycle_list = []
        duty_ratio_list = []

        duty_cycle_mean = 0
        duty_ratio_mean = 0
        i = 0
        time = np.array(self.logs['time'])

        while i < len(feetInContactBool):

            if feetInContactBool[i] == STANCE:
                T_stance = 0
                while i < len(feetInContactBool) and feetInContactBool[i] != SWING:
                    if i == 0:
                        time_passed = time[i]
                    else:
                        time_passed = time[i]-time[i-1]
                    T_stance = T_stance + time_passed
                    i += 1

                T_stance_list.append(T_stance)
            else:
                T_swing = 0
                while i < len(feetInContactBool) and feetInContactBool[i] != STANCE:
                    if i == 0:
                        time_passed = time[i]
                    else:
                        time_passed = time[i]-time[i-1]
                    T_swing = T_swing + time_passed
                    i += 1

                T_swing_list.append(T_swing)

        for j in range(min(len(T_swing_list), len(T_stance_list))):
            duty_cycle_list.append(T_swing_list[j] + T_stance_list[j])
            duty_ratio_list.append(T_stance_list[j] / T_swing_list[j])

        count1 = 0
        for i in range(len(duty_cycle_list)):
            if (i > 0) and (i < (len(duty_cycle_list) - 1)):  # to avoid init and final step
                duty_cycle_mean = duty_cycle_mean + duty_cycle_list[i]
                count1 = count1 + 1

        duty_cycle_mean = duty_cycle_mean / count1

        count1 = 0
        for i in range(len(duty_ratio_list)):
            if (i > 0) and (i < (len(duty_ratio_list) - 1)):  # to avoid init and final step
                duty_ratio_mean = duty_ratio_mean + duty_ratio_list[i]
                count1 = count1 + 1

        duty_ratio_mean = duty_ratio_mean / count1

        return duty_cycle_mean, duty_ratio_mean

    def duty_ratio(self, nb_step_evaluation):
        # duty cycle and ratio calculation
        feet_contact_bool = np.array(self.logs['feet_contact_bool'])

        feetInContactBool_save_leg0 = feet_contact_bool[:, 0]
        feetInContactBool_save_leg1 = feet_contact_bool[:, 1]
        feetInContactBool_save_leg2 = feet_contact_bool[:, 2]
        feetInContactBool_save_leg3 = feet_contact_bool[:, 3]
        duty_cycle_mean_leg0, duty_ratio_mean_leg0 = self.duty_cycle_and_ratio(feetInContactBool_save_leg0)
        duty_cycle_mean_leg1, duty_ratio_mean_leg1 = self.duty_cycle_and_ratio(feetInContactBool_save_leg1)
        duty_cycle_mean_leg2, duty_ratio_mean_leg2 = self.duty_cycle_and_ratio(feetInContactBool_save_leg2)
        duty_cycle_mean_leg3, duty_ratio_mean_leg3 = self.duty_cycle_and_ratio(feetInContactBool_save_leg3)

        duty_cycle_mean = (duty_cycle_mean_leg0 + duty_cycle_mean_leg1 + duty_cycle_mean_leg2 + duty_cycle_mean_leg3) / 4
        duty_ratio_mean = (duty_ratio_mean_leg0 + duty_ratio_mean_leg1 + duty_ratio_mean_leg2 + duty_ratio_mean_leg3) / 4
        print('Duty cycle mean: ', duty_cycle_mean, ' [s]')
        print('Duty ratio mean: ', duty_ratio_mean)
