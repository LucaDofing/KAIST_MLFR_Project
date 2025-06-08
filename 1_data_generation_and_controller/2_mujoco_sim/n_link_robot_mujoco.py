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

def extract_actuator_limits(model):
    """Extract actuator control limits from the MuJoCo model.
    
    Returns:
        np.ndarray: Array of torque limits for each actuator
    """
    if model.actuator_ctrlrange is None or len(model.actuator_ctrlrange) == 0:
        print("Warning: No actuator control ranges defined in XML. Using default 100.0 N⋅m")
        return np.full(model.nu, 100.0)
    
    # Extract upper limits from control ranges (assuming symmetric ranges)
    torque_limits = model.actuator_ctrlrange[:, 1]  # Upper bound of each actuator
    
    print(f"Extracted torque limits from XML: {torque_limits} N⋅m")
    return torque_limits

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
    cam.distance = 4
    cam.elevation = 0
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
            # Update visualization
            viewport = mujoco.MjrRect(0, 0, 1980, 1980)
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

def print_limits_example():
    """Print an example of how torque limits are now automatically enforced by MuJoCo."""
    print("\n" + "="*60)
    print("TORQUE LIMITS - AUTOMATICALLY ENFORCED BY MUJOCO")
    print("="*60)
    print("✅ MuJoCo AUTOMATICALLY enforces ctrlrange limits at the actuator level!")
    print("✅ No manual clipping needed - XML is the single source of truth!")
    print()
    print("Example XML generation with torque limits:")
    print("python3 1_xml_generator/1_generate_n_link_robot_xml.py \\")
    print("  --num_links 1 \\")
    print("  --link_mass 3.5 \\")
    print("  --torque_limit 25.0     # Defined in XML ctrlrange")
    print() 
    print("Example simulation (limits auto-enforced by MuJoCo):")
    print("python3 2_mujoco_sim/n_link_robot_mujoco.py \\")
    print("  --xml_path path/to/model.xml \\")  
    print("  --control_mode constant \\")
    print("  --constant_torque 50.0 \\       # Even exceeding XML limit!")
    print("  --log 1 \\                      # Enable logging")
    print("  --no-render                     # Faster execution")
    print()
    print("🔧 How it works:")
    print("  1. Controller can request ANY torque (e.g., 50.0 N⋅m)")
    print("  2. MuJoCo automatically clips actuator forces to XML ctrlrange (25.0 N⋅m)")
    print("  3. Robot physics uses the limited forces, not the requested torque")
    print()
    print("📊 Data logging captures:")
    print("  - Requested control: data.ctrl (may exceed limits)")
    print("  - Actual forces: data.actuator_force (automatically limited)")
    print("="*60 + "\n")

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
    parser.add_argument("--constant_torque", type=float, default=0.5,
                      help="Torque value for constant control mode")
    parser.add_argument("--target_angle", type=float, default=20.0,
                      help="Target angle in degrees for PD control")
    parser.add_argument("--initial_angle", type=float, default=0.0,
                      help="Initial angle in degrees for all joints")
    parser.add_argument("--kp", type=float, default=3.0,
                      help="Position gain for PD control")
    parser.add_argument("--kd", type=float, default=0.01,
                      help="Velocity gain for PD control")
    parser.add_argument("--no-render", action="store_true",
                      help="Run without visualization for faster data generation")
    parser.add_argument("--speedup", type=float, default=1.0,
                      help="Speedup factor for visualization (default: 1.0). A value of 2.0 means simulation plays twice as fast.")
    parser.add_argument("--log", type=int, default=0,
                      help="Enable data logging with 1, by default disabled ")
    
    # Show limits example option (--torque_limit argument removed!)
    parser.add_argument("--show_limits_example", action="store_true",
                      help="Show example usage of automatic torque limit extraction and exit")
    
    parser.add_argument("--robot_folder_name", type=str, default=None,
                      help="Robot folder name for organized data structure")
    parser.add_argument("--simulation_run_name", type=str, default=None, 
                      help="Simulation run folder name with timestamp")
    
    args = parser.parse_args()

    # Show example if requested
    if args.show_limits_example:
        print_limits_example()
        return

    # Load model and initialize simulation
    global model  # Needed for visualization
    model = load_model(args.xml_path)
    
    # Extract torque limits from the model (XML is now single source of truth!)
    torque_limits = extract_actuator_limits(model)
    
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
    
    # Initialize data logger with nested folder structure
    logger = DataLogger(
        robot_folder_name=args.robot_folder_name,
        simulation_run_name=args.simulation_run_name
    )
    logger.set_xml_model_path(args.xml_path)
    
    # Store simulation parameters - use extracted torque limits
    sim_params = {
        'control_mode': args.control_mode,
        'initial_angle': np.deg2rad(args.initial_angle),
        'target_angle': np.deg2rad(args.target_angle),
        'kp': args.kp,
        'kd': args.kd,
        'ki': 0.0,  # Default value for ki
        'torque_limit': float(torque_limits[0]) if len(torque_limits) > 0 else 100.0  # Use first limit for logging
    }
    
    # Print limit information - now from XML
    print(f"Motor Constraint (extracted from XML):")
    if len(torque_limits) == 1:
        print(f"  Torque Limit: ±{torque_limits[0]:.3f} N⋅m (AUTO-ENFORCED by MuJoCo ctrlrange)")
    else:
        print(f"  Torque Limits: {torque_limits} N⋅m (AUTO-ENFORCED by MuJoCo ctrlrange)")
    print(f"  Note: MuJoCo automatically limits actuator forces to ctrlrange values!")
    
    # Extract static properties with simulation parameters
    logger.extract_static_properties(model, sim_params)
    
    # Run simulation with or without rendering
    if args.no_render:
        run_simulation_no_rendering(model, data, controller, logger, args)
    else:
        run_simulation_with_rendering(model, data, controller, logger, args)
    sim_end_time = time.time()
    # Save logged data
    if args.log:
        # Set simulation time in metadata
        if args.no_render:
            logger.data["metadata"]["simulation_time"] = sim_end_time - sim_start_time
        else:
            logger.data["metadata"]["simulation_time"] = "Inf (rendering enabled)"
        logger.data["static_properties"]["controller_gains"] = {
            "kp": float(args.kp),
            "kd": float(args.kd),
            "ki": float(0.0)
        }
        logger.save_data()
        logger.save_xml_model(args.xml_path)
    
    print(f"Total Time: {sim_end_time-sim_start_time:.6f}")

if __name__ == "__main__":
    main() 