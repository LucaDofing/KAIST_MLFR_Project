# Parameter sweep configuration for XML and simulation

xml_params = {
    "n_links": [1],
    "link_length": [0.15],
    "link_radius": [0.01],
    "link_density": [10000],
    "joint_damping": [0],
    "joint_friction": [0.0],
}

sim_params = {
    "control_mode": ["pd"],
    "initial_angle": [90.0],
    "sim_time": [2.0],
    "target_angle": [40.0,0,180],
    "kp": [3.0],
    "kd": [0.01],
}
