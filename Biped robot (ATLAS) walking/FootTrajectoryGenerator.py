import numpy as np
import math
from matplotlib import pyplot as plt

class FootTrajectoryGenerator:
    def __init__(self, stepTime, doubleSupportTime, maximumStepHeight, alpha = 0.5, NumberOfFootSteps = 15):
        self.tStep_ = stepTime
        self.tDS_ = doubleSupportTime
        self.dt_ = 1/240
        self.alpha_ = alpha
        self.numberOfFootSteps = NumberOfFootSteps
        self.leftFirst_ = True
        self.height_ = maximumStepHeight
        pass

    def setFootPrints(self,foot_pose):
        self.footPose_ = foot_pose
        if self.footPose_[0,1] < 0:
            self.leftFirst_ = False # left foot swings first
        pass
    
    def generateTrajectory(self):
        self.lFoot_ = list("")
        self.rFoot_ = list("")

        if (self.leftFirst_):       # left foot swings first
            for step in range(1, self.numberOfFootSteps+1):
                if step % 2 == 0:   #  left is support, right swings
                    for time in np.arange(0.0, (1-self.alpha_) * self.tDS_, self.dt_):
                        self.lFoot_.append(self.footPose_[step])
                        self.rFoot_.append(self.footPose_[step - 1])
                        
                    coefs = self.polynomial(self.footPose_[step-1],self.footPose_[step+1], self.height_,self.tStep_-self.tDS_)
                    for time in np.arange(0.0, self.tStep_ - self.tDS_, self.dt_):
                        self.lFoot_.append(self.footPose_[step])
                        self.rFoot_.append(coefs[0] + coefs[1] * time + coefs[2] * time**2 + coefs[3] * time**3 + coefs[4] * time**4 + coefs[5] * time**5)
                        
                    for time in np.arange(0.0,(self.alpha_) * self.tDS_, self.dt_):
                        self.lFoot_.append(self.footPose_[step])
                        self.rFoot_.append(self.footPose_[step + 1])
                        
                else:
                    for time in np.arange(0.0, (1-self.alpha_) * self.tDS_, self.dt_):
                        self.lFoot_.append(self.footPose_[step - 1])
                        self.rFoot_.append(self.footPose_[step])
                        
                    coefs = self.polynomial(self.footPose_[step-1],self.footPose_[step+1], self.height_,self.tStep_-self.tDS_)
                    for time in np.arange(0.0, self.tStep_ - self.tDS_, self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(coefs[0] + coefs[1] * time + coefs[2] * time**2 + coefs[3] * time**3 + coefs[4] * time**4 + coefs[5] * time**5)
                        
                    for time in np.arange(0.0,self.alpha_ * self.tDS_,self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(self.footPose_[step + 1])
                        
        else:         # right foot swings first
            for step in range(1, self.stepCount+1):
                if step % 2 == 0:   
                    for time in np.arange(0.0, (1-self.alpha_ * self.tDS_), self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(self.footPose_[step - 1])
                        
                    coefs = self.polynomial(self.footPose_[step-1],self.footPose_[step+1], self.height_,self.tStep_-self.tDS_)
                    for time in np.arange(0.0, self.tStep_ - self.tDS_, self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(coefs[0] + coefs[1] * time + coefs[2] * time**2 + coefs[3] * time**3 + coefs[4] * time**4 + coefs[5] * time**5)
                        
                    for time in np.arange(0.0,(self.alpha) * self.tDS_,self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(self.footPose_[step + 1])
                        
                else:
                    for time in np.arange(0.0, (1-self.alpha_) * self.tDS_, self.dt_):
                        self.lFoot_.append(self.footPose_[step - 1])
                        self.rFoot_.append(self.footPose_[step])
                        
                    coefs = self.polynomial(self.footPose_[step-1],self.footPose_[step+1], self.height_,self.tStep_-self.tDS_)
                    for time in np.arange(0.0, self.tStep_ - self.tDS_, self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(coefs[0] + coefs[1] * time + coefs[2] * time**2 + coefs[3] * time**3 + coefs[4] * time**4 + coefs[5] * time**5)
                        
                    for time in np.arange(0.0,(self.alpha) * self.tDS_,self.dt_):
                        self.rFoot_.append(self.footPose_[step])
                        self.lFoot_.append(self.footPose_[step + 1])
                        
        pass

    def getRightFootTrajectory(self):
        return self.rFoot_

    def getLeftFootTrajectory(self):
        return self.lFoot_

    def polynomial(self,x0, xf, z_max, tf):
        ans = list("")
        ans.append(x0)
        ans.append(np.zeros(3))
        ans.append(np.zeros(3))
        ans.append(10/tf**3 * (xf - x0))
        ans.append(-15/tf**4 * (xf - x0))
        ans.append(6/tf**5 * (xf - x0))

        ans[0][2] = 0.0
        ans[1][2] = 0.0
        ans[2][2] = 16 * z_max / tf**2
        ans[3][2] = -32 * z_max / tf**3
        ans[4][2] = 16 * z_max / tf**4
        ans[5][2] = 0.0

        return ans 