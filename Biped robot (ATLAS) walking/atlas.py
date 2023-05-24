import pybullet as p
import time,math
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
atlas = p.loadURDF("atlas/atlas_v4_with_multisense.urdf", [-2,3,-0.5])
for i in range (p.getNumJoints(atlas)):
	p.setJointMotorControl2(atlas,i,p.POSITION_CONTROL,0)
	print(p.getJointInfo(atlas,i))


	p.loadURDF("plane.urdf",[0,0,-0])
	
p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=148, cameraPitch=-9, cameraTargetPosition=[0.36,5.3,-0.62])



p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

p.getCameraImage(320,200)#, renderer=p.ER_BULLET_HARDWARE_OPENGL )


t=0
p.setRealTimeSimulation(1)
while (1):
	p.setGravity(0,0,-10)
	time.sleep(0.01)
	t+=0.01
	
		
	
