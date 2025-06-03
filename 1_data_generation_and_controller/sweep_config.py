xml_params = {
    "n_links": [1],
    "link_length": [0.15,0.30],
    "link_radius": [0.01,0.05],
    "link_density": [100,500],
    "joint_damping": [0,0.2,0.4,0.6,0.8,1.0],
    "joint_friction": [0.0],
}

sim_params = {
    "control_mode": ["pd"],
    "initial_angle": [0,45,90,180,220],
    "sim_time": [4.0],
    "target_angle": [0,20,60,110,160],
    "kp": [0.5,3.0],
    "kd": [0.01],
}
