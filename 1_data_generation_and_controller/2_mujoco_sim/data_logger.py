import os
import json
import time
import numpy as np

class DataLogger:
    def __init__(self, save_dir="4_data/2_mujoco"):
        """Initialize the data logger.
        
        Args:
            save_dir (str): Directory to save the data files
        """
        self.save_dir = save_dir
        self.data = {
            "timestamps": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_torques": [],
            "target_positions": []
        }
        self.start_time = time.time()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def log_step(self, data, model, timestep):
        """Log data for a single simulation step.
        
        Args:
            data: MuJoCo simulation data
            model: MuJoCo model
            timestep (float): Current simulation time
        """
        self.data["timestamps"].append(timestep)
        self.data["joint_positions"].append(data.qpos.copy())
        self.data["joint_velocities"].append(data.qvel.copy())
        self.data["joint_torques"].append(data.ctrl.copy())
        
        # For now, using fixed target position
        target_pos = np.array([0.15, 0.15])
        self.data["target_positions"].append(target_pos)
    
    def save_data(self):
        """Save logged data to a JSON file."""
        # Convert all numpy arrays to lists for JSON serialization
        save_data = {
            "timestamps": self.data["timestamps"],
            "joint_positions": [pos.tolist() for pos in self.data["joint_positions"]],
            "joint_velocities": [vel.tolist() for vel in self.data["joint_velocities"]],
            "joint_torques": [torque.tolist() for torque in self.data["joint_torques"]],
            "target_positions": [pos.tolist() for pos in self.data["target_positions"]]
        }
        
        # Generate filename with timestamp
        filename = f"simulation_data_{int(time.time())}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save to file
        if True:
            with open(filepath, "w") as f:
                json.dump(save_data, f, indent=4)
            
        print(f"Data saved to {filepath}")
        
    def clear_data(self):
        """Clear all logged data."""
        self.data = {
            "timestamps": [],
            "joint_positions": [],
            "joint_velocities": [],
            "joint_torques": [],
            "target_positions": []
        }
        self.start_time = time.time() 