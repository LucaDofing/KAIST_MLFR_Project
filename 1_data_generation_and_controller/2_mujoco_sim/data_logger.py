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
                "solver": "Unknown"
            },
            "static_properties": {
                "nodes": [],
                "edge_index": []
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
        
    def extract_static_properties(self, model):
        """Extract static properties from the MuJoCo model."""
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
            
            # Calculate mass from density and volume
            if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # Volume of capsule = πr²h + (4/3)πr³
                r = geom.size[0]
                h = geom.size[1] * 2  # Total length
                volume = np.pi * r**2 * h + (4/3) * np.pi * r**3
            else:
                volume = 0.0
                
            # Get density from geom (default value if not specified)
            density = 1000.0  # Default density in kg/m³
            mass = volume * density
            
            # Get link length from geom
            if geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                length = geom.size[1] * 2  # Total length
            else:
                length = 0.0
                
            # Get damping and friction from joint defaults
            damping = model.dof_damping[joint_id]  # Use dof_damping instead of jnt_damping
            friction = 0.0  # Default friction value
            
            # Store link properties
            self.data["static_properties"]["nodes"].append({
                "mass": float(mass),
                "length": float(length),
                "radius": float(geom.size[0]),  # Add radius
                "damping": float(damping),
                "friction": float(friction),
                "inertia": body.inertia.tolist()  # Add inertia matrix
            })
            
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
        current_torque = data.ctrl[:model.nu].copy()
        
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
        """Save logged data to a JSON file."""
        # Generate filename with timestamp
        filename = f"simulation_data_{int(time.time())}.json"
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
                "solver": "Unknown"
            },
            "static_properties": {
                "nodes": [],
                "edge_index": []
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