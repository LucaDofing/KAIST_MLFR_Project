import mujoco
import numpy as np
import time
import argparse
import glfw
import os
import json

# Update View for Dynamic View
def load_model(xml_path):
    """Load the MuJoCo model from XML file."""
    return mujoco.MjModel.from_xml_path(xml_path)

def init_simulation(model):
    """Initialize the simulation data."""
    return mujoco.MjData(model)

def apply_random_actions(data, model):
    """Apply random actions to the actuators."""
    # Generate random actions between -1 and 1
    actions = np.random.uniform(-1, 1, model.nu)
    # Scale actions to reasonable torque limits (e.g., ±10 Nm)
    torques = actions * 10.0
    data.ctrl[:] = torques

def apply_constant_actions(data, model):
    """Apply constant torques to all actuators."""
    # Apply a constant torque of 5 Nm to all joints
    constant_torque = 5.0  # Nm
    data.ctrl[:] = constant_torque

def apply_pd_control(data, model):
    """Simple PD controller to move joints to 20 degrees."""
    # Convert 20 degrees to radians
    target_angle = np.deg2rad(10.0)
    
    # PD controller gains
    kp = 100.0  # Position gain
    kd = 10.0   # Velocity gain
    
    # Get current joint positions and velocities
    current_pos = data.qpos[:model.nu]  # First nu elements are actuated joints
    current_vel = data.qvel[:model.nu]
    
    # Calculate position and velocity errors
    pos_error = target_angle - current_pos
    vel_error = 0.0 - current_vel  # Target velocity is 0
    
    # Calculate control torques using PD control law
    torques = kp * pos_error + kd * vel_error
    
    # Apply torques
    data.ctrl[:] = torques

def save_simulation_data(data, model, timestep):
    """Save simulation data to a JSON file."""
    # Create the directory if it doesn't exist
    os.makedirs("4_data/2_mujoco", exist_ok=True)
    
    # Prepare data to save
    data_to_save = {
        "timestep": timestep,
        "joint_positions": data.qpos[:model.nu].tolist(),
        "joint_velocities": data.qvel[:model.nu].tolist(),
        "joint_torques": data.ctrl[:].tolist()
    }
    
    # Save to file
    filename = f"4_data/2_mujoco/simulation_data_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(data_to_save, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Run a simple MuJoCo simulation with the n_link_robot model")
    parser.add_argument("--xml_path", type=str, default="4_data/1_xml_models/n_link_robot.xml", help="Path to the MuJoCo XML file")
    parser.add_argument("--sim_time", type=float, default=10.0, help="Simulation time in seconds")
    parser.add_argument("--control_mode", type=str, default="pd", choices=["random", "constant", "pd"],
                      help="Control mode: random, constant, or pd")
    args = parser.parse_args()

    # Load the model and initialize simulation
    model = load_model(args.xml_path)
    data = init_simulation(model)

    # Print model information
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of sensors: {model.nsensor}")

    # Initialize visualization
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Create window
    window = glfw.create_window(1200, 900, "n_link_robot Simulation", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return

    glfw.make_context_current(window)

    # Initialize MuJoCo visualization
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    scene = mujoco.MjvScene(model, maxgeom=10000)

    # Set initial camera position
    cam.distance = 1.0  # Reduced from 3.0 to zoom in more
    cam.azimuth = 0.0   # Set to 0 for top-down view
    cam.elevation = -90.0  # Set to -90 for looking straight down

    # Run simulation
    start_time = time.time()
    while not glfw.window_should_close(window) and (time.time() - start_time) < args.sim_time:
        # Step the simulation
        mujoco.mj_step(model, data)
        
        # Save simulation data
        save_simulation_data(data, model, data.time)
        
        # Apply control based on selected mode
        if args.control_mode == "random":
            apply_random_actions(data, model)
        elif args.control_mode == "constant":
            apply_constant_actions(data, model)
        else:  # pd control
            apply_pd_control(data, model)
        
        # Update scene
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        
        # Render scene
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjr_render(viewport, scene, context)
        
        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        # Small sleep to prevent excessive CPU usage
        time.sleep(0.01)

    # Cleanup
    glfw.terminate()

if __name__ == "__main__":
    main() 