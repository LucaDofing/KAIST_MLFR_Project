xml_params = {
    "n_links": [1],
    "link_length": [0.15],
    "link_radius": [0.01],
    "link_density": [500,1000],
    "joint_damping": [0,0.2,0.4,0.6,0.8,1.0],
    "joint_friction": [0.0],
}

sim_params = {
    "control_mode": ["pd"],
    "initial_angle": [90.0],
    "sim_time": [4.0],
    "target_angle": [45.0],
    "kp": [0.5,3.0],
    "kd": [0.01],
}