import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


class DCMTrajectoryGenerator:
    def __init__(self, pelvisHeight, stepTiming, doubleSupportTime):
        self.CoMHeight = pelvisHeight  # We assume that CoM and pelvis are the same point
        self.stepDuration = stepTiming
        self.dsTime = doubleSupportTime
        self.timeStep = 1 / 240  # We select this value for the timestep(dt) for discretization of the trajectory. The 240 Hz is the default numerical solving frequency of the pybullet. Therefore we select this value for DCM trajectory generation discretization.
        self.numberOfSamplesPerSecond = 240  # Number of sampling of the trajectory in each second
        self.numberOfSteps = 14  # This is the desired number of steps for walking
        self.alpha = 0.5  # We have 0<alpha<1 that is used for double support simulation
        self.DCM = list("")
        self.gravityAcceleration = 9.81
        self.omega = 0.85* math.sqrt(self.gravityAcceleration / self.CoMHeight)  # Omega is a constant value and is called natural frequency of linear inverted pendulum
        pass

    def getDCMTrajectory(self):
        self.findFinalDCMPositionsForEachStep()  # compute discrete DCM positions
        self.planDCMForSingleSupport()  # Plan preliminary DCM trajectory (DCM without considering double support)
        self.findBoundryConditionsOfDCMDoubleSupport()  # Find the boundary conditions for double support
        self.embedDoubleSupportToDCMTrajectory()  # Do interpolation for double support and embed double support phase trajectory to the preliminary trajectory
        return self.DCM

    def getCoMTrajectory(self, com_ini):
        # This class generates the CoM trajectory by integration of CoM velocity(that has been found by the DCM values)
        self.CoM = np.zeros_like(self.DCM)
        self.CoMDot = np.zeros_like(self.DCM)
        self.CoM[0] = com_ini
        self.CoMDot[0] = 0
        for kk in range(0, self.CoM.shape[0] - 1):
            self.CoMDot[kk + 1] = self.omega * (self.DCM[kk] - self.CoM[kk])  # Equation (3)
            self.CoM[kk + 1] = self.CoM[kk] + self.timeStep * self.CoMDot[kk]  # Simple euler numerical integration
            self.CoM[kk + 1][2] = self.CoMHeight
        return self.CoM

    def setCoP(self, CoP):
        self.CoP = CoP  # setting CoP positions. Note: The CoP has an offset with footprint positions
        pass

    def setFootPrints(self, footPrints):
        self.footPrints = footPrints  # setting footprint positions. Note: The footprint has an offset with CoP positions

    def findFinalDCMPositionsForEachStep(
            self):  # Finding Final(=initial for previous, refer to equation 8) dcm for a step
        self.DCMForEndOfStep = np.copy(self.CoP)  # initialization for having same shape
        self.DCMForEndOfStep[-1] = self.CoP[-1]  # capturability constraint(3rd item of jupyter notebook steps for DCM motion planning section)

        for index in range(np.size(self.CoP, 0) - 2, -1, -1):
            self.DCMForEndOfStep[index] = self.CoP[index+1] + (self.DCMForEndOfStep[index+1] - self.CoP[index+1]) * np.exp(-self.omega * self.stepDuration)  # equation 7
        pass

    def calculateCoPTrajectory(self):
        self.DCMVelocity = np.zeros_like(self.DCM)
        self.CoPTrajectory = np.zeros_like(self.DCM)
        self.DCMVelocity[0] = 0
        self.CoPTrajectory[0] = self.CoP[0]
        for kk in range(0, self.CoM.shape[0] - 1):
            self.DCMVelocity[kk+1] = (
                        self.DCM[kk+1] - self.DCM[kk])/self.timeStep  # Numerical differentiation for solving DCM Velocity
            self.CoPTrajectory[kk+1] = self.DCM[kk+1] - self.DCMVelocity[kk+1]/self.omega  # Use equation (10) to find CoP by having DCM and DCM Velocity


    def planDCMForSingleSupport(self):  # The output of this function is a DCM vector with a size of (int(self.numberOfSamplesPerSecond* self.stepDuration * self.CoP.shape[0])) that is number of sample points for whole time of walking
        for iter in range(int(self.numberOfSamplesPerSecond * self.stepDuration * self.CoP.shape[0])):  # We iterate on the whole simulation control cycles:
            time = iter/self.numberOfSamplesPerSecond  # Finding the time of a corresponding control cycle
            i = iter // (int(self.stepDuration * self.numberOfSamplesPerSecond))  # Finding the number of corresponding step of walking
            t = time - (self.stepDuration * i)  # The “internal” step time t is reset at the beginning of each step

            self.DCM.append(self.CoP[i] + (self.DCMForEndOfStep[i] - self.CoP[i])*np.exp(self.omega * (t - self.stepDuration)))  # Use equation (9) for finding the DCM trajectory
        self.DCM = np.array(self.DCM)

    def findBoundryConditionsOfDCMDoubleSupport(self):

        self.initialDCMForDS = np.zeros((np.size(self.CoP, 0), 3))
        self.finalDCMForDS = np.zeros((np.size(self.CoP, 0), 3))
        self.initialDCMVelocityForDS = np.zeros((np.size(self.CoP, 0), 3))
        self.finalDCMVelocityForDS = np.zeros((np.size(self.CoP, 0), 3))

        for stepNumber in range(np.size(self.CoP, 0)):
            # Boundary conditions of double support for the first step(equation 11b and 12b)
            if stepNumber == 0:
                self.initialDCMForDS[stepNumber] = self.DCM[stepNumber]  # eq (11b)
                self.finalDCMForDS[stepNumber] = self.CoP[stepNumber] + (self.DCM[stepNumber] - self.CoP[stepNumber]) * np.exp(self.omega * (1 - self.alpha) * self.dsTime)  # eq (12b)
                self.initialDCMVelocityForDS[stepNumber] = self.omega * (self.initialDCMForDS[stepNumber] - self.CoP[stepNumber])  # eq (4)
                self.finalDCMVelocityForDS[stepNumber] = self.omega * (self.finalDCMForDS[stepNumber] - self.CoP[stepNumber])  # eq (4)

            # Boundary conditions of double support for all steps except first step (equation 11 and 12)
            else:
                self.initialDCMForDS[stepNumber] = self.CoP[stepNumber - 1] + (self.DCMForEndOfStep[stepNumber - 1] - self.CoP[stepNumber - 1])*np.exp(-self.omega * self.alpha * self.dsTime)  # eq (11)
                self.finalDCMForDS[stepNumber] = self.CoP[stepNumber] + (self.DCMForEndOfStep[stepNumber - 1] - self.CoP[stepNumber])* np.exp(self.omega * (1 - self.alpha) * self.dsTime)  # eq (12)
                self.initialDCMVelocityForDS[stepNumber] = self.omega * (self.initialDCMForDS[stepNumber] - self.CoP[stepNumber])  # eq (4)
                self.finalDCMVelocityForDS[stepNumber] = self.omega * (self.finalDCMForDS[stepNumber] - self.CoP[stepNumber])  # eq (4)

    def doInterpolationForDoubleSupport(self, initialDCMForDS, finalDCMForDS, initialDCMVelocityForDS, finalDCMVelocityForDS, dsTime):
        # The implementation of equation (15) of Jupyter Notebook
        a = (2 * initialDCMForDS) / (dsTime ** 3) + (initialDCMVelocityForDS) / (
                    dsTime ** 2) - (2 * finalDCMForDS) / (dsTime ** 3) + (finalDCMVelocityForDS) / (dsTime ** 2)  # first element of P matrix
        b = - (3 * initialDCMForDS) / (dsTime ** 2) - (
                    2 * initialDCMVelocityForDS) / dsTime + (3 * finalDCMForDS) / (dsTime ** 2) - finalDCMVelocityForDS / dsTime  # second element of P matrix
        c = initialDCMVelocityForDS  # third element of P matrix
        d = initialDCMForDS  # fourth element of P matrix
        return a, b, c, d  # a b c and are the elements of the P in equation (15)

    # Calculate and replace DCM position for double support with the corresponding time window of preliminary single support phase
    def embedDoubleSupportToDCMTrajectory(self):
        self.doubleSupportInterpolationCoefficients = list('')

        for stepNumber in range(np.size(self.CoP, 0)):
            if (stepNumber == 0):
                a, b, c, d = self.doInterpolationForDoubleSupport(self.initialDCMForDS[stepNumber], self.finalDCMForDS[stepNumber],
                    self.initialDCMVelocityForDS[stepNumber], self.finalDCMVelocityForDS[stepNumber], self.dsTime * (1 - self.alpha))
                self.doubleSupportInterpolationCoefficients.append([a, b, c, d])  # Create a vector of DCM Coeffient by using the doInterpolationForDoubleSupport function. Note that the double support duration for first step is not the same as other steps
            else:
                a, b, c, d = self.doInterpolationForDoubleSupport(self.initialDCMForDS[stepNumber], self.finalDCMForDS[stepNumber],
                    self.initialDCMVelocityForDS[stepNumber], self.finalDCMVelocityForDS[stepNumber],self.dsTime)

                self.doubleSupportInterpolationCoefficients.append([a, b, c, d])

        # In the following part we will find the list of double support trajectories for all steps of walking
        self.listOfDoubleSupportTrajectories = list('')
        for stepNumber in range(np.size(self.CoP, 0)):
            if (stepNumber == 0):  # notice double support duration is not the same as other steps

                a, b, c, d = self.doubleSupportInterpolationCoefficients[stepNumber]
                doubleSupportTrajectory = np.zeros((int((1 - self.alpha) * self.dsTime / self.timeStep), 3))

                for t in range(int(self.dsTime * (1 - self.alpha) * self.numberOfSamplesPerSecond)):
                    doubleSupportTrajectory[t] = a * ((t * self.timeStep) ** 3) + b * ((t * self.timeStep) ** 2) + c * ( t * self.timeStep) + d  # use equation 16 (only the DCM position ecomponent)

                self.listOfDoubleSupportTrajectories.append(doubleSupportTrajectory)

            else:
                a, b, c, d = self.doubleSupportInterpolationCoefficients[stepNumber]
                doubleSupportTrajectory = np.zeros((int(self.dsTime / self.timeStep), 3))
                for t in range(int(self.dsTime * self.numberOfSamplesPerSecond)):
                    doubleSupportTrajectory[t] = a * ((t * self.timeStep) ** 3) + b * ((t * self.timeStep) ** 2) + c * (
                                t * self.timeStep) + d  # use equation 16 (only the DCM position component)
                self.listOfDoubleSupportTrajectories.append(doubleSupportTrajectory)

        # In the following part we will replace the double support trajectories for the corresponding double support time-window  in the preliminary DCM trajectory
        # First we put preliminary DCM trajectory into a new array and in th following we will replace the double support part
        DCMCompleteTrajectory = np.array(self.DCM)

        for stepNumber in range(self.CoP.shape[0]):
            index_alpha = int(self.alpha * self.dsTime * self.numberOfSamplesPerSecond)
            index_step = int(self.stepDuration * self.numberOfSamplesPerSecond * stepNumber)
            if stepNumber == 0:
                # fill the corresponding interval for DCM index for double support part
                DCMCompleteTrajectory[0:index_alpha] = self.listOfDoubleSupportTrajectories[stepNumber]
            else:
                # fill the corresponding interval for DCM index for double support part
                DCMCompleteTrajectory[index_step - index_alpha:index_step + index_alpha] = self.listOfDoubleSupportTrajectories[stepNumber]

        self.DCM = DCMCompleteTrajectory
        temp = np.array(self.DCM)