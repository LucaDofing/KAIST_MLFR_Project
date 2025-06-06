import os
import json
import time
import numpy as np
import mujoco

class DataLogger:
    def __init__(self, save_dir="4_data/2_mujoco"):
        """Initialize the data logger.
        
        Args:
            save_dir (str): Directory to save the data files
        """
        self.save_dir = save_dir
        self.data = {
            "metadata": {
                "num_links": 0,
                "num_steps": 0,
                "dt": 0.0,
                "gravity": [0.0, 0.0, 0.0],
                "solver": "Unknown",
                "simulation_time": None  # Will be set when saving data
            },
            "static_properties": {
                "nodes": [],
                "edge_index": [],
                "controller_gains": {
                    "kp": None,
                    "kd": None,
                    "ki": None
                },
                "initial_angle_deg": None,
                "target_angle_deg": None
            },
            "time_series": {
                "theta": [],
                "omega": [],
                "alpha": [],
                "torque": []
            }
        }
        self.prev_omega = None
        self.start_time = time.time()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def _get_next_simulation_number(self):
        """Get the next simulation number from a counter file."""
        counter_file = os.path.join(self.save_dir, "simulation_counter.txt")
        
        # Create counter file if it doesn't exist
        if not os.path.exists(counter_file):
            with open(counter_file, "w") as f:
                f.write("0")
        
        # Read and increment counter
        with open(counter_file, "r+") as f:
            # Use file locking to prevent race conditions
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            
            counter = int(f.read().strip())
            counter += 1
            
            # Write back the new counter
            f.seek(0)
            f.write(str(counter))
            f.truncate()
            
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
        return counter
        
    def extract_static_properties(self, model, sim_params=None):
        """Extract static properties from the MuJoCo model.
        
        Args:
            model: MuJoCo model
            sim_params (dict, optional): Dictionary containing simulation parameters
        """
        # Initialize data for mass matrix calculation
        data = mujoco.MjData(model)
        mujoco.mj_step1(model, data)
        
        # Get number of links (excluding the root and fingertip)
        num_links = model.nbody - 2  # Subtract root and fingertip
        
        # Extract all metadata from model options
        self.data["metadata"]["gravity"] = model.opt.gravity.tolist()
        self.data["metadata"]["dt"] = model.opt.timestep
        self.data["metadata"]["num_links"] = num_links
        
        # Get solver from model options
        solver_map = {
            mujoco.mjtIntegrator.mjINT_EULER: "Euler",
            mujoco.mjtIntegrator.mjINT_RK4: "RK4",
            mujoco.mjtIntegrator.mjINT_IMPLICIT: "Implicit",
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "ImplicitFast"
        }
        self.data["metadata"]["solver"] = solver_map.get(model.opt.integrator, "Unknown")
        
        # Extract fingertip mass (last body in the model)
        fingertip_mass = 0.0
        if model.nbody > num_links + 1:  # Check if fingertip exists
            fingertip_body_id = model.nbody - 1  # Last body is fingertip
            fingertip_mass = float(model.body_mass[fingertip_body_id])
        
        # Extract controller gains and angles from sim_params
        if sim_params is not None:
            # Store controller gains
            self.data["static_properties"]["controller_gains"] = {
                "kp": float(sim_params.get('kp', 0.0)),
                "kd": float(sim_params.get('kd', 0.0)),
                "ki": float(sim_params.get('ki', 0.0))
            }
            
            # Store initial and target angles in degrees
            initial_angle_rad = sim_params.get('initial_angle', 0.0)
            target_angle_rad = sim_params.get('target_angle', 0.0)
            self.data["static_properties"]["initial_angle_deg"] = float(np.rad2deg(initial_angle_rad))
            self.data["static_properties"]["target_angle_deg"] = float(np.rad2deg(target_angle_rad))
            
            # Store torque limit only (most realistic motor constraint)
            self.data["static_properties"]["limits"] = {
                "torque_limit_nm": float(sim_params.get('torque_limit', 100.0))
            }
        
        # ========================================================================
        # MASS MATRIX INERTIA EXTRACTION (MuJoCo's actual values)
        # ========================================================================
        
        # Get the full mass matrix from MuJoCo - this is what the physics engine actually uses
        mass_matrix = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        
        # For a single revolute joint, the (0,0) element is the rotational inertia
        actual_inertia_from_mass_matrix = float(mass_matrix[0, 0])
        
        # Extract properties for each link
        for i in range(num_links):
            body_id = i + 1  # Skip root body
            body = model.body(body_id)
            
            # Get joint properties
            joint_id = model.body_jntadr[body_id]
            joint = model.joint(joint_id)
            
            # Get geom properties
            geom_id = model.body_geomadr[body_id]
            geom = model.geom(geom_id)
            
            # Get mass and basic properties
            mass = model.body_mass[body_id]
            
            # Get link length from geom
            if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                length = geom.size[1] * 2  # Total length
            else:
                length = 0.0
                
            # Get damping from joint defaults
            damping = model.dof_damping[joint_id]
            friction = 0.0  # Default friction value
            
            # Store clean node properties with only mass matrix inertia
            node_properties = {
                "mass": float(mass),
                "fingertip_mass": fingertip_mass,
                "length": float(length),
                "radius": float(geom.size[0]),
                "damping": float(damping),
                "friction": float(friction),
                "inertia": float(actual_inertia_from_mass_matrix),  # MuJoCo's actual inertia from mass matrix
            }
            
            # Add limits if sim_params exist
            if sim_params is not None:
                node_properties["torque_limit_nm"] = float(sim_params.get('torque_limit', 100.0))
            
            self.data["static_properties"]["nodes"].append(node_properties)
            
            # Print clean summary
            print(f"\nLink {i} properties:")
            print(f"  Mass: {mass:.3f} kg")
            print(f"  Fingertip mass: {fingertip_mass:.3f} kg")
            print(f"  Length: {length:.3f} m")
            print(f"  Radius: {geom.size[0]:.3f} m")
            print(f"  🎯 Inertia (mass matrix): {actual_inertia_from_mass_matrix:.8f} kg⋅m²")
            
            # Add edge to kinematic chain
            if i > 0:
                self.data["static_properties"]["edge_index"].append([i-1, i])
        
    def log_step(self, data, model, timestep):
        """Log data for a single simulation step.
        
        Args:
            data: MuJoCo simulation data
            model: MuJoCo model
            timestep (float): Current simulation time
        """
        # Extract current state
        current_theta = data.qpos[:model.nu].copy()
        current_omega = data.qvel[:model.nu].copy()
        
        # Log ACTUAL applied torques (limited by MuJoCo ctrlrange) instead of commanded torques
        current_torque = data.actuator_force[:model.nu].copy()
        
        # Calculate acceleration (alpha) using finite differences
        if self.prev_omega is not None:
            current_alpha = (current_omega - self.prev_omega) / model.opt.timestep
        else:
            current_alpha = np.zeros_like(current_omega)
        
        # Store actual values
        self.data["time_series"]["theta"].append(current_theta.tolist())
        self.data["time_series"]["omega"].append(current_omega.tolist())
        self.data["time_series"]["alpha"].append(current_alpha.tolist())
        self.data["time_series"]["torque"].append(current_torque.tolist())
        
        # Update previous values
        self.prev_omega = current_omega
        
        # Update step count
        self.data["metadata"]["num_steps"] += 1
    
    def save_data(self):
        """Save logged data to a JSON file with a descriptive name based on parameters."""
        # Get next simulation number
        sim_number = self._get_next_simulation_number()
        
        # Get parameters for filename
        num_links = self.data["metadata"]["num_links"]
        initial_angle = self.data["static_properties"]["initial_angle_deg"]
        target_angle = self.data["static_properties"]["target_angle_deg"]
        kp = self.data["static_properties"]["controller_gains"]["kp"]
        kd = self.data["static_properties"]["controller_gains"]["kd"]
        
        # Get damping from first link (assuming all links have same damping)
        damping = self.data["static_properties"]["nodes"][0]["damping"] if self.data["static_properties"]["nodes"] else 0.0
        
        # Create descriptive filename with simulation ID
        filename = f"sim_{sim_number:03d}_n_link_{num_links}_init_{initial_angle:.1f}_target_{target_angle:.1f}_kp_{kp:.1f}_kd_{kd:.3f}_damping_{damping:.3f}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=4)
            
        print(f"Data saved to {filepath}")
        
    def clear_data(self):
        """Clear all logged data."""
        self.data = {
            "metadata": {
                "num_links": 0,
                "num_steps": 0,
                "dt": 0.0,
                "gravity": [0.0, 0.0, 0.0],
                "solver": "Unknown",
                "simulation_time": None  # Will be set when saving data
            },
            "static_properties": {
                "nodes": [],
                "edge_index": [],
                "controller_gains": {
                    "kp": None,
                    "kd": None,
                    "ki": None
                },
                "initial_angle_deg": None,
                "target_angle_deg": None
            },
            "time_series": {
                "theta": [],
                "omega": [],
                "alpha": [],
                "torque": []
            }
        }
        self.prev_omega = None
        self.start_time = time.time()

    def set_sweep_params_static(self, sweep_params: dict):
        """Store sweep parameters in static_properties.
        
        Args:
            sweep_params (dict): Dictionary containing all sweep parameters including:
                - initial_conditions: dict with theta and omega
                - target_angles: list of target angles
                - controller_params: dict with type, kp, kd, ki
                - simulation_params: dict with duration and dt
        """
        # Update initial conditions
        if "initial_conditions" in sweep_params:
            self.data["static_properties"]["initial_conditions"] = sweep_params["initial_conditions"]
        
        # Update target angles
        if "target_angles" in sweep_params:
            self.data["static_properties"]["target_angles"] = sweep_params["target_angles"]
        
        # Update controller parameters
        if "controller_params" in sweep_params:
            self.data["static_properties"]["controller_params"] = sweep_params["controller_params"]
        
        # Update simulation parameters
        if "simulation_params" in sweep_params:
            self.data["static_properties"]["simulation_params"] = sweep_params["simulation_params"]
        
        # Store all sweep parameters
        self.data["static_properties"]["sweep_params"] = sweep_params 

def get_actual_mujoco_inertia(model):
    """Get the actual inertia that MuJoCo uses for dynamics."""
    data = mujoco.MjData(model)
    mujoco.mj_step1(model, data)  # Initialize
    
    # Get the mass matrix
    mass_matrix = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass_matrix, data.qM)
    
    # For a single revolute joint, the [0,0] element is the rotational inertia
    return mass_matrix[0, 0] 