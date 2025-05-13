import mujoco
import numpy as np
import time
import argparse
import glfw
import os
from data_logger import DataLogger
from controllers import create_controller

def load_model(xml_path):
    """Load the MuJoCo model from XML file."""
    return mujoco.MjModel.from_xml_path(xml_path)

def init_simulation(model, initial_angle=0.0):
    """Initialize the simulation data.
    
    Args:
        model: MuJoCo model
        initial_angle (float): Initial angle in radians for all joints
    """
    data = mujoco.MjData(model)
    # Set all joint angles to the same initial value
    data.qpos[:model.nu] = initial_angle
    return data

def init_visualization():
    """Initialize GLFW and MuJoCo visualization."""
    if not glfw.init():
        return None, None, None, None, None
    
    # Create window
    window = glfw.create_window(1200, 900, "n_link_robot Simulation", None, None)
    if not window:
        glfw.terminate()
        return None, None, None, None, None
    
    glfw.make_context_current(window)
    
    # Initialize MuJoCo visualization
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
    
    # Set camera viewpoint
    cam.distance = 1.0
    cam.elevation = -20
    cam.azimuth = 90
    cam.lookat[0] = 0.0  # x
    cam.lookat[1] = 0.0  # y
    cam.lookat[2] = 0.0  # z
    
    return window, cam, opt, scene, context

def rad2deg(rad):
    """Convert radians to degrees."""
    return rad * 180.0 / np.pi



def run_simulation_with_rendering(model, data, controller, logger, args):
    """Run simulation with visualization."""
    # Initialize visualization
    window, cam, opt, scene, context = init_visualization()
    if window is None:
        print("Failed to initialize visualization")
        return
    
    # Print header
    print("\nTime(s) | Joint1(deg) | Joint2(deg) | Vel1(deg/s) | Vel2(deg/s) | Torque1(Nm) | Torque2(Nm)")
    print("-" * 100)

    # Calculate timing parameters
    RENDER_INTERVAL = 1.0 / 60.0  # Fixed 60 Hz refresh rate
    DEFAULT_TIMESTEP = 0.01  # Standard MuJoCo timestep
    actual_speedup = args.speedup * (model.opt.timestep / DEFAULT_TIMESTEP)  # Adjust speedup for timestep
    
    sim_start_time = time.time()  # Reference point for timing calculations
    next_render_time = sim_start_time  # Next time we should render a frame

    print(f"Physics timestep: {model.opt.timestep:.6f}s, Adjusted speedup: {actual_speedup:.1f}x")

    # Run simulation
    while not glfw.window_should_close(window) and data.time < args.sim_time:
        # Get control action and step physics until next render time
        current_time = time.time()
        
        # Run physics steps until we reach the next visualization point
        target_sim_time = (current_time - sim_start_time) * args.speedup  # Use original speedup for visualization
        
        while data.time < min(target_sim_time, args.sim_time):
            action = controller.get_action()
            data.ctrl[:] = action
            mujoco.mj_step(model, data)
            logger.log_step(data, model, data.time)
        
        # Render if it's time
        if current_time >= next_render_time:
            # Print state (synchronized with rendering)
            
            # Update visualization
            viewport = mujoco.MjrRect(0, 0, 1200, 900)
            mujoco.mjv_updateScene(model, data, opt, None, cam,
                                  mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            mujoco.mjr_render(viewport, scene, context)
            
            # Swap buffers and poll events
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            # Calculate next render time
            next_render_time = current_time + RENDER_INTERVAL
    
    # Cleanup
    glfw.terminate()

def run_simulation_no_rendering(model, data, controller, logger, args):
    """Run simulation without visualization for faster data generation."""
    print("Running simulation without visualization...")
    print(f"Simulation time: {args.sim_time} seconds")
    print(f"Physics timestep: {model.opt.timestep} seconds")
    
    # Calculate progress update interval based on speedup
    progress_interval = 1.0  # Print every second of simulation time
    next_print_time = progress_interval
    
    # Run simulation
    while data.time < args.sim_time:
        # Get control action
        action = controller.get_action()
        
        # Apply action and step simulation
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        
        # Log every physics step
        logger.log_step(data, model, data.time)
        
        # Print progress at regular intervals of simulation time
        if data.time >= next_print_time:
            print(f"Simulation progress: {data.time:.1f}/{args.sim_time:.1f} seconds")
            next_print_time += progress_interval

def main():
    sim_start_time = time.time()
    parser = argparse.ArgumentParser(description="Run a simple MuJoCo simulation with the n_link_robot model")
    parser.add_argument("--xml_path", type=str, default="4_data/1_xml_models/n_link_robot.xml",
                      help="Path to the MuJoCo XML file")
    parser.add_argument("--sim_time", type=float, default=20.0,
                      help="Simulation time in seconds")
    parser.add_argument("--control_mode", type=str, default="pd",
                      choices=["random", "constant", "pd"],
                      help="Control mode: random, constant, or pd")
    parser.add_argument("--constant_torque", type=float, default=5.0,
                      help="Torque value for constant control mode")
    parser.add_argument("--target_angle", type=float, default=10.0,
                      help="Target angle in degrees for PD control")
    parser.add_argument("--initial_angle", type=float, default=0.0,
                      help="Initial angle in degrees for all joints")
    parser.add_argument("--kp", type=float, default=100.0,
                      help="Position gain for PD control")
    parser.add_argument("--kd", type=float, default=10.0,
                      help="Velocity gain for PD control")
    parser.add_argument("--no-render", action="store_true",
                      help="Run without visualization for faster data generation")
    parser.add_argument("--speedup", type=float, default=1.0,
                      help="Speedup factor for visualization (default: 1.0). A value of 2.0 means simulation plays twice as fast.")
    parser.add_argument("--log", type=int, default=0,
                      help="Enable data logging with 1, by default disabled ")
    args = parser.parse_args()

    # Load model and initialize simulation
    global model  # Needed for visualization
    model = load_model(args.xml_path)
    data = init_simulation(model, np.deg2rad(args.initial_angle))
        
    # Create controller-specific parameter dictionaries
    controller_params = {}
    if args.control_mode == "constant":
        controller_params = {"constant_torque": args.constant_torque}
    elif args.control_mode == "pd":
        controller_params = {
            "target_angle": np.deg2rad(args.target_angle),
            "kp": args.kp,
            "kd": args.kd
        }
    
    # Initialize controller
    controller = create_controller(args.control_mode, model, data, **controller_params)
    
    # Initialize data logger
    logger = DataLogger()
    logger.extract_static_properties(model)  # Extract static properties from the model
    
    # Run simulation with or without rendering
    if args.no_render:
        run_simulation_no_rendering(model, data, controller, logger, args)
    else:
        run_simulation_with_rendering(model, data, controller, logger, args)

    # Save logged data
    if args.log:
        logger.save_data()
    sim_end_time = time.time()
    print(f"Total Time: {sim_end_time-sim_start_time:.6f}")



if __name__ == "__main__":
    main() 