All Units in the simulation are SI units

n_link_robot.xml , has been used for all simulations. 


1_link_pd_controller.json has a d gain of 0, therefore is only a proportional controller. 
There was some bug i need to fix with the controller, but i think it is not relevant for our initial tesing of the GNN. 

Maybe you can compare the values of ur simulation with dynamic equations, with the mujoco one. At least for the simulation with no torque.
Initial conditions and dynamic properties you can extract from the .json file
