import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
import os

def create_n_link_robot_xml(
    # Default Parameters
    n_links=1,
    link_length=0.15,          # Length of each link
    link_radius=0.01,          # Radius of the links
    joint_range=(-200.0, 200.0),   # Range of motion for the second joint
    target_pos=(0.15, 0.15, 0.0),   # Fixed position for the target
    motor_gear=1.0,          # Gear ratio for the motors
    motor_range=(-100.0, 100.0),   # Control range for the motors
    gravity=(0, 0, -9.81),     # Gravity vector (z-down)
    timestep=0.01,             # Simulation timestep
    integrator="implicit",          # Integration method
    link_density=1000.0,       # Density of the links in kg/m³
    joint_damping=0.1,         # Damping coefficient for joints
    joint_friction=0.0,        # Friction coefficient for joints
    joint_armature=0.0,        # Armature inertia for joints
    joint_stiffness=0.0,       # Joint stiffness
    joint_springref=0.0,       # Joint spring reference position
    colors={
        "ground": "0.9 0.9 0.9 1",
        "sides": "0.8 0.3 0.5 1",
        "root": "0.8 0.3 0.5 1",
        "links": "0.0 0.3 0.5 1",
        "fingertip": "0.0 0.7 0.5 1",
        "target": "0.8 0.1 0.1 1"
    }

    ## Default n_link_robot parameters
):
    # Create the root element
    root = ET.Element("mujoco", model="n_link_robot")
    
    # Add compiler options
    compiler = ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="true")
    
    # Add asset section for textures and materials
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.6 0.6 0.6", rgb2="0.3 0.3 0.3", width="512", height="512")
    
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
    
    # Add camera
    ET.SubElement(worldbody, "camera", 
                 name="track", mode="trackcom", 
                 pos="0.3 -0.3 0.3", xyaxes="1 0 0 0 1 1",
                 target="body0")
    
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

        # Add geom
        ET.SubElement(prev_body, "geom",
                     fromto=f"0 0 0 0 0 {link_length}",
                     name=f"link{i}", rgba=colors["links"],
                     size=str(link_radius), type="capsule",
                     density=str(link_density))

        # Add actuator for this joint
        ET.SubElement(actuator, "motor",
                     ctrllimited="true",
                     ctrlrange=f"{motor_range[0]} {motor_range[1]}",
                     gear=str(motor_gear), joint=f"joint{i}")

        # If this is the last link, add the fingertip here
        if i == n_links - 1:
            fingertip = ET.SubElement(prev_body, "body", name="fingertip", pos=f"0 0 {link_length}")
            ET.SubElement(fingertip, "geom",
                         contype="0", name="fingertip",
                         pos="0 0 0", rgba=colors["fingertip"],
                         size=str(link_radius), type="sphere",
                         density=str(link_density))
        else:
            # Otherwise, create the next body for the next link
            next_body = ET.SubElement(prev_body, "body", name=f"body{i+1}", pos=f"0 0 {link_length}")
            prev_body = next_body
    
    # Add fixed target at a different position
    ET.SubElement(worldbody, "geom", 
                 conaffinity="0", contype="0", name="target", 
                 pos=f"0.15 0.0 0.15", 
                 rgba=colors["target"], 
                 size=".009", type="sphere")
    
    # Convert to string with pretty printing
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
    
    return xmlstr

def save_n_link_robot_xml(xml_content, filename="n_link_robot.xml"):
    # Get the absolute path to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    xml_dir = os.path.join(project_root, "1_data_generation_and_controller", "4_data", "1_xml_models")
    
    # Create the directory if it doesn't exist
    os.makedirs(xml_dir, exist_ok=True)
    # Save the file in the correct location
    
    with open(os.path.join(xml_dir, filename), "w") as f:
        f.write(xml_content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate n_link_robot.xml with a specified number of links")
    parser.add_argument("--num_links", type=int, default=1, help="Number of links in the n_link_robot robot (default: 1)")
    args = parser.parse_args()

    xml_content = create_n_link_robot_xml(n_links=args.num_links)
    save_n_link_robot_xml(xml_content)
    print(f"Generated n_link_robot.xml with {args.num_links} links")
    