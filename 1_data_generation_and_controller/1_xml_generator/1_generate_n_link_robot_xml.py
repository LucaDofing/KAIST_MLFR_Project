import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
import os
import math

# Import default parameters
try:
    from parameters import get_default_params, print_default_config
    DEFAULT_PARAMS = get_default_params()
except ImportError:
    print("Warning: parameters.py not found. Using hardcoded defaults.")
    DEFAULT_PARAMS = {
        "n_links": 1,
        "link_length": 0.15,
        "link_radius": 0.01,
        "link_mass": 1.5,
        "fingertip_mass": None,
        "joint_damping": 0.1,
        "joint_friction": 0.0,
        "joint_armature": 0.0,
        "joint_stiffness": 0.0,
        "joint_springref": 0.0,
        "joint_range": (-200.0, 200.0),
        "torque_limit": 25.0,
        "motor_gear": 1.0,
        "timestep": 0.01,
        "integrator": "implicit",
        "gravity": (0, 0, -9.81),
        "output_dir": ".",
        "output_name": "n_link_robot.xml",
        "target_pos": (0.15, 0.15, 0.0),
        "colors": {
            "ground": "0.9 0.9 0.9 1",
            "sides": "0.8 0.3 0.5 1",
            "root": "0.8 0.3 0.5 1",
            "links": "0.0 0.3 0.5 1",
            "fingertip": "0.0 0.7 0.5 1",
            "target": "0.8 0.1 0.1 1",
            "x_axis": "1.0 0.0 0.0 1",
            "y_axis": "0.0 1.0 0.0 1",
            "z_axis": "0.0 0.0 1.0 1"
        },
        "camera_pos": (0.0, -0.8, 0.6),
        "camera_xyaxes": "1 0 0 0 0.8 0.6",
        "camera_target": "body0",
        "camera_mode": "trackcom"
    }

def get_xml_models_dir():
    """
    Get the absolute path to the XML models directory.
    This ensures files are saved to the correct location regardless of where the script is called from.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the target directory: ../4_data/1_xml_models/
    xml_models_dir = os.path.join(script_dir, "..", "4_data", "1_xml_models")
    # Return the absolute path
    return os.path.abspath(xml_models_dir)

def create_n_link_robot_xml(**kwargs):
    """
    Create an n-link robot XML model for MuJoCo.
    
    Parameters are loaded in this order (later overrides earlier):
    1. Default values from parameters.py
    2. Keyword arguments passed to this function
    
    This allows main.py and command line arguments to override defaults.
    """
    # Start with default parameters
    params = DEFAULT_PARAMS.copy()
    
    # Override with any provided kwargs
    params.update(kwargs)
    
    # Extract individual parameters for readability
    n_links = params["n_links"]
    link_length = params["link_length"]
    link_radius = params["link_radius"]
    link_mass = params["link_mass"]
    fingertip_mass = params["fingertip_mass"]
    joint_damping = params["joint_damping"]
    joint_friction = params["joint_friction"]
    joint_armature = params["joint_armature"]
    joint_stiffness = params["joint_stiffness"]
    joint_springref = params["joint_springref"]
    joint_range = params["joint_range"]
    torque_limit = params["torque_limit"]
    motor_gear = params["motor_gear"]
    timestep = params["timestep"]
    integrator = params["integrator"]
    gravity = params["gravity"]
    colors = params["colors"]

    # Calculate fingertip mass if not specified
    if fingertip_mass is None:
        # Auto-calculate fingertip mass using same density as link
        # Link geometry: capsule = cylinder + 2 hemispheres
        cylinder_volume = math.pi * link_radius**2 * link_length
        hemisphere_volume = (4/3) * math.pi * link_radius**3
        link_volume = cylinder_volume + hemisphere_volume
        link_density = link_mass / link_volume
        
        # Fingertip geometry: sphere
        fingertip_volume = (4/3) * math.pi * link_radius**3
        fingertip_mass = link_density * fingertip_volume
        
        print(f"Auto-calculated fingertip mass: {fingertip_mass:.6f} kg")
    else:
        print(f"Using specified fingertip mass: {fingertip_mass:.6f} kg")
    
    # Set motor range based on torque limit
    motor_range = (-torque_limit, torque_limit)
    
    print(f"Link specifications:")
    print(f"  Mass: {link_mass:.3f} kg")
    print(f"  Length: {link_length:.3f} m, Radius: {link_radius:.3f} m")
    print(f"Fingertip specifications:")
    print(f"  Mass: {fingertip_mass:.6f} kg")
    print(f"  Radius: {link_radius:.3f} m")
    print(f"Motor specifications:")
    print(f"  Torque limit: ±{torque_limit:.1f} N⋅m")
    
    # Create the root element
    root = ET.Element("mujoco", model="n_link_robot")
    
    # Add compiler options
    compiler = ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="true")
    
    # Add asset section for textures and materials
    asset = ET.SubElement(root, "asset")
    # Change background to light grey
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.7 0.7 0.7", rgb2="0.7 0.7 0.7", width="512", height="512")
    
    # Add visual options
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "map", znear="0.001")
    ET.SubElement(visual, "scale", forcewidth="0.1")
    
    # Add default settings
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", 
                 armature=str(joint_armature), 
                 damping=str(joint_damping), 
                 frictionloss=str(joint_friction),
                 stiffness=str(joint_stiffness),
                 springref=str(joint_springref),
                 limited="true")
    ET.SubElement(default, "geom", contype="0", friction="1 0.1 0.1", rgba="1 1 1 1")
    
    # Add simulation options
    option = ET.SubElement(root, "option", 
                          gravity=f"{gravity[0]} {gravity[1]} {gravity[2]}", 
                          integrator=integrator, 
                          timestep=str(timestep))
    
    # Add worldbody
    worldbody = ET.SubElement(root, "worldbody")
    
    # Add coordinate system visualization
    axis_length = 0.2  # Length of coordinate axes
    axis_radius = 0.002  # Radius of coordinate axes
    
    # X-axis (red)
    ET.SubElement(worldbody, "geom",
                 type="cylinder",
                 fromto=f"0 0 0 {axis_length} 0 0",
                 size=str(axis_radius),
                 rgba=colors["x_axis"])
    
    # Y-axis (green)
    ET.SubElement(worldbody, "geom",
                 type="cylinder",
                 fromto=f"0 0 0 0 {axis_length} 0",
                 size=str(axis_radius),
                 rgba=colors["y_axis"])
    
    # Z-axis (blue)
    ET.SubElement(worldbody, "geom",
                 type="cylinder",
                 fromto=f"0 0 0 0 0 {axis_length}",
                 size=str(axis_radius),
                 rgba=colors["z_axis"])
    
    # Add camera
    camera_pos = kwargs.get("camera_pos", (0.0, -1.0, 0.8))
    camera_xyaxes = kwargs.get("camera_xyaxes", "1 0 0 0 0.8 0.6")
    camera_target = kwargs.get("camera_target", "body0")
    camera_mode = kwargs.get("camera_mode", "trackcom")
    
    ET.SubElement(worldbody, "camera", 
                 name="track", mode=camera_mode, 
                 pos=f"{camera_pos[0]} {camera_pos[1]} {camera_pos[2]}", 
                 xyaxes=camera_xyaxes,
                 target=camera_target)
    
    # Add light
    ET.SubElement(worldbody, "light",
                 cutoff="100",
                 diffuse="1 1 1",
                 dir="-0 0 -1.3",
                 directional="true",
                 exponent="1",
                 pos="0.3 0.3 1.3",
                 specular=".1 .1 .1")
    
    # Add root
    ET.SubElement(worldbody, "geom", 
                 conaffinity="0", contype="0", 
                 fromto="0 0 0 0 0 0.02", name="root", 
                 rgba=colors["root"], size=".011", type="cylinder")
    
    # Create actuator section
    actuator = ET.SubElement(root, "actuator")
    
    # Initialize the first body at the base
    prev_body = ET.SubElement(worldbody, "body", name="body0", pos="0 0 0.01")
    
    # Create the chain of links
    for i in range(n_links):
        # Add joint
        joint_attribs = {
            "axis": "0 1 0",  # Changed to y-axis for rotation
            "name": f"joint{i}",
            "pos": "0 0 0",
            "type": "hinge",
            "damping": str(joint_damping),
            "frictionloss": str(joint_friction),
            "armature": str(joint_armature),
            "stiffness": str(joint_stiffness),
            "springref": str(joint_springref)
        }
        if i == 0:
            joint_attribs["limited"] = "false"
        else:
            joint_attribs["limited"] = "true"
            joint_attribs["range"] = f"{joint_range[0]} {joint_range[1]}"
        ET.SubElement(prev_body, "joint", **joint_attribs)

        # Add geom - using MASS instead of DENSITY
        ET.SubElement(prev_body, "geom",
                     fromto=f"0 0 0 0 0 {link_length}",
                     name=f"link{i}", rgba=colors["links"],
                     size=str(link_radius), type="capsule",
                     mass=str(link_mass))  # CHANGED: using mass instead of density

        # Add actuator for this joint
        ET.SubElement(actuator, "motor",
                     ctrllimited="true",
                     ctrlrange=f"{motor_range[0]} {motor_range[1]}",
                     gear=str(motor_gear), joint=f"joint{i}")

        # If this is the last link, add the fingertip here
        if i == n_links - 1:
            fingertip = ET.SubElement(prev_body, "body", name="fingertip", pos=f"0 0 {link_length}")
            # Use specific fingertip mass - using MASS instead of DENSITY
            ET.SubElement(fingertip, "geom",
                         contype="0", name="fingertip",
                         pos="0 0 0", rgba=colors["fingertip"],
                         size=str(link_radius), type="sphere",
                         mass=str(fingertip_mass))  # CHANGED: using mass instead of density
        else:
            # Otherwise, create the next body for the next link
            next_body = ET.SubElement(prev_body, "body", name=f"body{i+1}", pos=f"0 0 {link_length}")
            prev_body = next_body
    
    # Convert to string with pretty printing
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
    
    return xmlstr

def main():
    parser = argparse.ArgumentParser(
        description="Generate an n-link robot XML model for MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate using all default parameters
  python3 1_generate_n_link_robot_xml.py

  # View current default configuration
  python3 parameters.py

  # Generate 2-link robot with custom parameters
  python3 1_generate_n_link_robot_xml.py --num_links 2 --link_mass 3.0

  # Generate industrial robot
  python3 1_generate_n_link_robot_xml.py --link_length 0.5 --torque_limit 100.0
        """
    )
    
    # Robot Structure Parameters
    parser.add_argument("--num_links", type=int, default=DEFAULT_PARAMS["n_links"],
                       help=f"Number of links in the robot (default: {DEFAULT_PARAMS['n_links']})")
    parser.add_argument("--link_length", type=float, default=DEFAULT_PARAMS["link_length"],
                       help=f"Length of each link in meters (default: {DEFAULT_PARAMS['link_length']})")
    parser.add_argument("--link_radius", type=float, default=DEFAULT_PARAMS["link_radius"],
                       help=f"Radius of each link in meters (default: {DEFAULT_PARAMS['link_radius']})")
    parser.add_argument("--link_mass", type=float, default=DEFAULT_PARAMS["link_mass"],
                       help=f"Mass of each link in kg (default: {DEFAULT_PARAMS['link_mass']})")
    parser.add_argument("--fingertip_mass", type=float, default=DEFAULT_PARAMS["fingertip_mass"],
                       help="Mass of fingertip/gripper in kg (default: auto-calculated from link density)")
    
    # Joint Parameters
    parser.add_argument("--joint_damping", type=float, default=DEFAULT_PARAMS["joint_damping"],
                       help=f"Joint damping coefficient (default: {DEFAULT_PARAMS['joint_damping']})")
    parser.add_argument("--joint_friction", type=float, default=DEFAULT_PARAMS["joint_friction"],
                       help=f"Joint friction coefficient (default: {DEFAULT_PARAMS['joint_friction']})")
    parser.add_argument("--joint_armature", type=float, default=DEFAULT_PARAMS["joint_armature"],
                       help=f"Joint armature inertia (default: {DEFAULT_PARAMS['joint_armature']})")
    parser.add_argument("--joint_stiffness", type=float, default=DEFAULT_PARAMS["joint_stiffness"],
                       help=f"Joint stiffness (default: {DEFAULT_PARAMS['joint_stiffness']})")
    parser.add_argument("--joint_springref", type=float, default=DEFAULT_PARAMS["joint_springref"],
                       help=f"Joint spring reference position (default: {DEFAULT_PARAMS['joint_springref']})")
    parser.add_argument("--joint_range", type=float, nargs=2, default=DEFAULT_PARAMS["joint_range"],
                       metavar=('MIN', 'MAX'),
                       help=f"Joint range in degrees (min max) (default: {DEFAULT_PARAMS['joint_range']})")
    
    # Actuator Parameters
    parser.add_argument("--torque_limit", type=float, default=DEFAULT_PARAMS["torque_limit"],
                       help=f"Maximum torque in N⋅m (default: {DEFAULT_PARAMS['torque_limit']})")
    parser.add_argument("--motor_gear", type=float, default=DEFAULT_PARAMS["motor_gear"],
                       help=f"Motor gear ratio (default: {DEFAULT_PARAMS['motor_gear']})")
    
    # Physics Parameters
    parser.add_argument("--timestep", type=float, default=DEFAULT_PARAMS["timestep"],
                       help=f"Simulation timestep in seconds (default: {DEFAULT_PARAMS['timestep']})")
    parser.add_argument("--integrator", type=str, default=DEFAULT_PARAMS["integrator"],
                       choices=["implicit", "explicit"],
                       help=f"Integration method (default: {DEFAULT_PARAMS['integrator']})")
    parser.add_argument("--gravity", type=float, nargs=3, default=DEFAULT_PARAMS["gravity"],
                       metavar=('X', 'Y', 'Z'),
                       help=f"Gravity vector (x y z) (default: {DEFAULT_PARAMS['gravity']})")
    
    # Output Parameters
    parser.add_argument("--output_dir", type=str, default=DEFAULT_PARAMS["output_dir"],
                       help=f"Directory to save the generated XML file (default: {DEFAULT_PARAMS['output_dir']})")
    parser.add_argument("--output_name", type=str, default=DEFAULT_PARAMS["output_name"],
                       help=f"Filename for the XML file (default: {DEFAULT_PARAMS['output_name']})")
    
    # Camera Parameters
    parser.add_argument("--camera_pos", type=float, nargs=3, default=DEFAULT_PARAMS["camera_pos"],
                       metavar=('X', 'Y', 'Z'),
                       help=f"Camera position (x y z) (default: {DEFAULT_PARAMS['camera_pos']})")
    parser.add_argument("--camera_xyaxes", type=str, default=DEFAULT_PARAMS["camera_xyaxes"],
                       help=f"Camera orientation axes (default: {DEFAULT_PARAMS['camera_xyaxes']})")
    parser.add_argument("--camera_target", type=str, default=DEFAULT_PARAMS["camera_target"],
                       help=f"Camera target body (default: {DEFAULT_PARAMS['camera_target']})")
    parser.add_argument("--camera_mode", type=str, default=DEFAULT_PARAMS["camera_mode"],
                       choices=["trackcom", "fixed", "track"],
                       help=f"Camera tracking mode (default: {DEFAULT_PARAMS['camera_mode']})")
    
    # Special options
    parser.add_argument("--show_config", action="store_true",
                       help="Show current default configuration and exit")
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        print_default_config()
        return
    
    # Get the absolute path to XML models directory (ignores args.output_dir for consistency)
    xml_models_dir = get_xml_models_dir()
    
    # Create output directory if it doesn't exist
    os.makedirs(xml_models_dir, exist_ok=True)
    
    # Convert parsed arguments to parameter dictionary
    xml_params = {
        "n_links": args.num_links,
        "link_length": args.link_length,
        "link_radius": args.link_radius,
        "link_mass": args.link_mass,
        "fingertip_mass": args.fingertip_mass,
        "joint_damping": args.joint_damping,
        "joint_friction": args.joint_friction,
        "joint_armature": args.joint_armature,
        "joint_stiffness": args.joint_stiffness,
        "joint_springref": args.joint_springref,
        "joint_range": tuple(args.joint_range),
        "torque_limit": args.torque_limit,
        "motor_gear": args.motor_gear,
        "timestep": args.timestep,
        "integrator": args.integrator,
        "gravity": tuple(args.gravity),
        "camera_pos": tuple(args.camera_pos),
        "camera_xyaxes": args.camera_xyaxes,
        "camera_target": args.camera_target,
        "camera_mode": args.camera_mode
    }
    
    # Generate XML using parameters
    xml_str = create_n_link_robot_xml(**xml_params)
    
    # Save to file using absolute path
    output_file = os.path.join(xml_models_dir, args.output_name)
    with open(output_file, "w") as f:
        f.write(xml_str)
    
    print(f"\n✅ Generated XML model with {args.num_links} links")
    print(f"📁 Saved to: {output_file}")
    print(f"💡 View default config with: python3 parameters.py")

if __name__ == "__main__":
    main() 