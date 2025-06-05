# Robot model parameters - define the physical robot (one robot per sweep)
robot_model_params = {
    "n_links": 1,
    "link_length": 0.40,
    "link_radius": 0.075,
    "link_mass": 3.0,  # Using mass instead of density
    "joint_damping": 0.3,
    "torque_limit": 0.05,  # Torque limit in N⋅m
}

# Simulation parameters - vary these to create the dataset
simulation_sweep_params = {
    "control_mode": ["pd"],
    "initial_angle": [0.0, 30.0, 60.0, 90.0],  # degrees
    "target_angle": [45.0, 120.0],  # degrees
    "kp": [30000.0],
    "kd": [0.010],
    "sim_time": [4.0],  # Fixed simulation time
}