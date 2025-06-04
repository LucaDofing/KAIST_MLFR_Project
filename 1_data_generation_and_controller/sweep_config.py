xml_params = {
    "n_links": [1],
    "link_length": [0.15,0.3],
    "link_radius": [0.01,0.05],
    "link_density": [500,1000,1500],
    "joint_damping": [0.4],
    "joint_friction": [0.0],
}

sim_params = {
    "control_mode": ["pd"],
    "initial_angle": [90.0,180,30],
    "sim_time": [4.0],
    "target_angle": [45.0],
    "kp": [0.5,3.0],
    "kd": [0.01],
}