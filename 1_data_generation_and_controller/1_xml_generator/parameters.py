# Default XML Generation Parameters
# This file contains all default parameters for XML robot model generation.
# These can be overridden by command line arguments or by main.py parameter sweeps.

# Robot Structure Parameters
ROBOT_STRUCTURE = {
    "n_links": 1,
    "link_length": 0.3,        # Length of each link in meters
    "link_radius": 0.075,        # Radius of each link in meters
    "link_mass": 3,           # Mass of each link in kg
    "fingertip_mass": 0.5,     # Mass of fingertip/gripper in kg (None = auto-calculate)
}

# Joint Parameters
JOINT_PARAMS = {
    "joint_damping": 0.5,       # Joint damping coefficient
    "joint_friction": 0.0,      # Joint friction coefficient
    "joint_armature": 0.0,      # Joint armature inertia
    "joint_stiffness": 0.0,     # Joint stiffness
    "joint_springref": 0.0,     # Joint spring reference position
    "joint_range": (-200.0, 200.0),  # Joint range in degrees (min, max)
}

# Actuator Parameters
ACTUATOR_PARAMS = {
    "torque_limit": 5000,       # Maximum torque in N⋅m
    "motor_gear": 1.0,          # Motor gear ratio
}

# Physics Parameters
PHYSICS_PARAMS = {
    "timestep": 0.01,           # Simulation timestep in seconds
    "integrator": "implicit",   # Integration method: implicit|explicit
    "gravity": (0, 0, -9.81),   # Gravity vector (x, y, z)
}

# Output Parameters
OUTPUT_PARAMS = {
    "output_dir": "../4_data/1_xml_models/",  # Output directory for XML file
    "output_name": "n_link_robot.xml",  # Default filename for XML
}

# Visual Parameters (not typically overridden)
VISUAL_PARAMS = {
    "colors": {
        "ground": "0.9 0.9 0.9 1",
        "sides": "0.8 0.3 0.5 1", 
        "root": "0.8 0.3 0.5 1",
        "links": "0.0 0.3 0.5 1",
        "fingertip": "0.0 0.7 0.5 1",
        "x_axis": "1.0 0.0 0.0 1",  # Red for X axis
        "y_axis": "0.0 1.0 0.0 1",  # Green for Y axis
        "z_axis": "0.0 0.0 1.0 1"   # Blue for Z axis
    }
}

# Camera Parameters - Added to center viewpoint on coordinate system
CAMERA_PARAMS = {
    "camera_pos": (0.0, -0.0, 0.0),    # Moved further back and higher for zoomed out view
    "camera_xyaxes": "1 0 0 0 0.8 0.6", # Adjusted orientation for better view of origin
    "camera_target": "body0",           # Keep tracking the robot base
    "camera_mode": "trackcom"           # Track center of mass mode
}

def get_default_params():
    """
    Get all default parameters as a single dictionary.
    
    Returns:
        dict: Combined dictionary of all default parameters
    """
    params = {}
    params.update(ROBOT_STRUCTURE)
    params.update(JOINT_PARAMS)
    params.update(ACTUATOR_PARAMS)
    params.update(PHYSICS_PARAMS)
    params.update(OUTPUT_PARAMS)
    params.update(VISUAL_PARAMS)
    params.update(CAMERA_PARAMS)
    return params

def print_default_config():
    """Print the current default configuration."""
    print("="*60)
    print("DEFAULT XML GENERATION PARAMETERS")
    print("="*60)
    
    print("\n🔧 Robot Structure:")
    for key, value in ROBOT_STRUCTURE.items():
        print(f"  {key}: {value}")
    
    print("\n⚙️  Joint Parameters:")
    for key, value in JOINT_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n🔌 Actuator Parameters:")
    for key, value in ACTUATOR_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n🌍 Physics Parameters:")
    for key, value in PHYSICS_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n📁 Output Parameters:")
    for key, value in OUTPUT_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n📹 Camera Parameters:")
    for key, value in CAMERA_PARAMS.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("💡 These can be overridden via command line arguments")
    print("💡 Or automatically by main.py parameter sweeps")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_default_config() 