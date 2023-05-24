# This repository is the implementation of two robots in simulation

## Biped robot

The first project control a biped robot, ATLAS, using inverted pendulum to control the feet path and control. 

## Quadruped robot
Quadruped robot walking is the implementation of a quadruped robot, similar to spot. It is trained using reinforcement learning to make it robust to a rough environment. 
In addition, a controller using Central Pattern Generator is investigated. CPGs are neural networks located in the spine of vertebrates and controlled by descending paths from the brain.
For robotics applications, CPGs are frequently modeled as coupled oscillators. This type of models are very useful for locomotion control as they can generate smooth rhythmic pattern which are stable against state perturbations, thanks to their inherent limit cycle behaviour.
To setup the gait, e.g. Trot, Walk, Bound, Pace, Pronk and Gallop, the offset between diverse limb must be parametrised. <br>
Then, PD controllers are developped. Once using a cartesian controler and once using a joint controller. In comparison to CPGs, PD controllers allow to better control and direct the quadruped. <br>

To reinforce and make the controller more robust, the outputs of the open-loop CPG controller are augmented using Virtual Model Control. VMC is a motion control framework that uses simulations of virtual components
to generate desired joint torques by creating the illusion that the simulated components are connected to the real robot. This method has proved to be useful for posture control on unperceived rough terrain locomotion with dynamic gait.
Finally, reinforcement learning is used to create a robust controller. Reinforcement learning is about an autonomous agent taking suitable actions to maximise rewards in a particular environment. The basic idea behind this method is
to conduct an important number of simulations (in the order of millions for complex environment) to learn a control policy.

Further details are available in the report.
