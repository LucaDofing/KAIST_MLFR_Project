# Robot model parameters - define the physical robot (one robot per sweep)
robot_model_params = {
    "n_links": 1,
    "link_length": 0.40,
    "link_radius": 0.075,
    "link_mass": 7.0,  # Using mass instead of density
    "fingertip_mass": 0.5,  # Mass of fingertip/gripper in kg (None = auto-calculate)
    "joint_damping": 0.7,  # You'll run multiple sweeps with different values
    "torque_limit": 20.0,  # Torque limit in N⋅m
}

# Simulation parameters - vary these to create the dataset
simulation_sweep_params = {
    "control_mode": ["pd"],
    "initial_angle": [0.0, 30.0, 60.0],  # More initial angles
    "target_angle": [45.0, 90.0],  # More target angles
    "kp": [10.0, 20.0, 30.0, 40.0, 50.0],  # More kp values
    "kd": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],  # More kd values
    "sim_time": [4.0, 6.0, 8.0, 10.0],  # Fixed simulation time
}
