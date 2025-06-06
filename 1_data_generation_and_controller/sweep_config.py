# Robot model parameters - define the physical robot (one robot per sweep)
robot_model_params = {
    "n_links": 1,
    "link_length": 0.40,
    "link_radius": 0.075,
    "link_mass": 7.0,  # Using mass instead of density
    "fingertip_mass": 0.5,  # Mass of fingertip/gripper in kg (None = auto-calculate)
    "joint_damping": 0.7,
    "torque_limit": 20.0,  # Torque limit in N⋅m
}

# Simulation parameters - vary these to create the dataset
simulation_sweep_params = {
    "control_mode": ["pd"],
    "initial_angle": [0.0],  # degrees
    "target_angle": [45.0],  # degrees
    "kp": [30.0,10],
    "kd": [0.010],
    "sim_time": [4.0],  # Fixed simulation time
}
