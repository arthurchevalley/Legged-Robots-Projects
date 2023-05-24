import numpy as np
import math

class RobotUtils:
    def __init__(self):
        self.config_ = np.zeros(12)
        self.pelvis_ = 0.089 
        self.shank_ = 0.422
        self.hip_ =  0.374
        pass

    def doInverseKinematics(self, pelvisP, pelvisR, leftP, leftR, rightP, rightR):
        self.config_[0:6] = self.solveIK(pelvisP, pelvisR,rightP, rightR, self.hip_,-self.pelvis_, self.shank_)
        self.config_[6:12] = self.solveIK(pelvisP, pelvisR,leftP, leftR, self.hip_, self.pelvis_, self.shank_)
        return self.config_[0:12]

    def Rroll(self, phi):
        R = np.eye(3)
        R[1,1] = np.cos(phi)
        R[1,2] = -np.sin(phi)
        R[2,1] = np.sin(phi)
        R[2,2] = np.cos(phi)
        return R

    def Rpitch(self, theta):
        R = np.eye(3)
        R[0,0] = np.cos(theta)
        R[0,2] = np.sin(theta)
        R[2,0] = -np.sin(theta)
        R[2,2] = np.cos(theta)
        return R

    def solveIK(self, p1,R1,p7,R7, A , d, B):
        D = np.array([0,d,0])
        r = np.matmul(R7.T , (p1 + np.matmul(R1 , D) - p7))
        C = np.sqrt(r[0]**2 + r[1]**2 +r[2]**2)
        c5 = (C**2 - A**2 - B**2) / (2*A*B)
        if c5 >= 1:
            q5 = 0.0
        elif c5 <= -1:
            q5 = np.pi
        else:
            q5 = np.arccos(c5)    
        q6a = np.arcsin((A/C)*np.sin(np.pi-q5)) 
        q7 = np.arctan2(r[1],r[2]) 
        if q7 > np.pi/2:
            q7 -= np.pi
        elif q7 < -np.pi/2:
            q7 += np.pi
        q6 = -np.arctan2(r[0], np.sign(r[2]) * np.sqrt(r[1]**2 + r[2]**2)) - q6a 
        R = np.matmul(R1.T , np.matmul(R7 , np.matmul(self.Rroll(-q7) , self.Rpitch(-q6-q5))))
        q2 = np.arctan2(-R[0,1], R[1,1]) 
        q3 = np.arctan2(R[2,1], -R[0,1] * np.sin(q2) + R[1,1] * np.cos(q2)) 
        q4 = np.arctan2(-R[2,0], R[2,2]) 
                        
        return([q2,q3,q4,q5,q6,q7])
