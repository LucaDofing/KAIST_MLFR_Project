import os
import json
import itertools
import subprocess
from typing import Dict, List, Any, Union
import numpy as np

class ParameterSweep:
    def __init__(self, base_dir: str = "1_data_generation_and_controller"):
        """Initialize the parameter sweep.
        
        Args:
            base_dir (str): Base directory for the project
        """
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.xml_dir = os.path.join(self.base_dir, "4_data/1_xml_models")
        self.data_dir = os.path.join(self.base_dir, "4_data/2_mujoco")
        
        # Create directories if they don't exist
        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define parameter ranges
        self.xml_params = {
            "n_links": [1, 2, 3],  # Number of links
            "link_length": [0.1, 0.15, 0.2],  # Length of each link
            "link_radius": [0.005, 0.01, 0.015],  # Radius of the links
            "link_density": [500.0, 1000.0, 1500.0],  # Density of the links
            "joint_damping": [0.05, 0.1, 0.2],  # Damping coefficient
            "joint_friction": [0.0, 0.1, 0.2],  # Friction coefficient
        }
        
        self.sim_params = {
            "control_mode": ["random", "constant", "pd"],  # Control modes
            "initial_angle": [0.0, 45.0, 90.0],  # Initial angles in degrees
            "sim_time": [10.0],  # Simulation time in seconds
            "constant_torque": [0.5, 1.0, 2.0],  # For constant controller
            "target_angle": [20.0, 45.0, 90.0],  # For PD controller
            "kp": [1.0, 3.0, 5.0],  # For PD controller
            "kd": [0.01, 0.05, 0.1],  # For PD controller
        }
        
    def generate_xml(self, params: Dict[str, Any]) -> str:
        """Generate XML file with given parameters.
        
        Args:
            params (Dict[str, Any]): XML generation parameters
            
        Returns:
            str: Path to the generated XML file
        """
        # Create filename based on parameters
        filename = f"n_link_robot_{'_'.join(f'{k}_{v}' for k, v in params.items())}.xml"
        xml_path = os.path.join(self.xml_dir, filename)
        
        # Build command
        cmd = [
            "python", os.path.join(self.base_dir, "1_xml_generator/1_generate_n_link_robot_xml.py"),
            "--num_links", str(params["n_links"]),
            "--output_dir", self.xml_dir
        ]
        
        # Run XML generation
        subprocess.run(cmd, check=True)
        
        # Rename the generated file
        default_xml = os.path.join(self.xml_dir, "n_link_robot.xml")
        if os.path.exists(default_xml):
            os.rename(default_xml, xml_path)
        
        return xml_path
    
    def run_simulation(self, xml_path: str, params: Dict[str, Any]) -> str:
        """Run simulation with given parameters.
        
        Args:
            xml_path (str): Path to the XML file
            params (Dict[str, Any]): Simulation parameters
            
        Returns:
            str: Path to the generated data file
        """
        # Build command
        cmd = [
            "python", os.path.join(self.base_dir, "2_mujoco_sim/n_link_robot_mujoco.py"),
            "--xml_path", xml_path,
            "--sim_time", str(params["sim_time"]),
            "--control_mode", params["control_mode"],
            "--initial_angle", str(params["initial_angle"]),
            "--no-render",  # Disable rendering for faster data generation
            "--log", "1"  # Enable data logging
        ]
        
        # Add controller-specific parameters
        if params["control_mode"] == "constant":
            cmd.extend(["--constant_torque", str(params["constant_torque"])])
        elif params["control_mode"] == "pd":
            cmd.extend([
                "--target_angle", str(params["target_angle"]),
                "--kp", str(params["kp"]),
                "--kd", str(params["kd"])
            ])
        
        # Run simulation
        subprocess.run(cmd, check=True)
        
        # Find the most recent data file
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        if not data_files:
            raise RuntimeError("No data file was generated")
        
        # Get the most recent file
        latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(self.data_dir, x)))
        return os.path.join(self.data_dir, latest_file)
    
    def run_sweep(self, xml_param_combinations: List[Dict[str, Any]] = None,
                 sim_param_combinations: List[Dict[str, Any]] = None):
        """Run parameter sweep.
        
        Args:
            xml_param_combinations (List[Dict[str, Any]], optional): Specific XML parameter combinations to use
            sim_param_combinations (List[Dict[str, Any]], optional): Specific simulation parameter combinations to use
        """
        # Generate parameter combinations if not provided
        if xml_param_combinations is None:
            xml_param_combinations = self._generate_param_combinations(self.xml_params)
        if sim_param_combinations is None:
            sim_param_combinations = self._generate_param_combinations(self.sim_params)
        
        # Run sweep
        results = []
        for xml_params in xml_param_combinations:
            # Generate XML
            xml_path = self.generate_xml(xml_params)
            
            for sim_params in sim_param_combinations:
                # Skip incompatible combinations
                if not self._is_valid_combination(xml_params, sim_params):
                    continue
                
                # Run simulation
                data_path = self.run_simulation(xml_path, sim_params)
                
                # Store results
                results.append({
                    "xml_params": xml_params,
                    "sim_params": sim_params,
                    "xml_path": xml_path,
                    "data_path": data_path
                })
        
        # Save sweep results
        self._save_sweep_results(results)
    
    def _generate_param_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all possible combinations of parameters.
        
        Args:
            params (Dict[str, List[Any]]): Parameter ranges
            
        Returns:
            List[Dict[str, Any]]: List of parameter combinations
        """
        keys = params.keys()
        values = params.values()
        combinations = []
        
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _is_valid_combination(self, xml_params: Dict[str, Any], sim_params: Dict[str, Any]) -> bool:
        """Check if a combination of XML and simulation parameters is valid.
        
        Args:
            xml_params (Dict[str, Any]): XML parameters
            sim_params (Dict[str, Any]): Simulation parameters
            
        Returns:
            bool: True if the combination is valid
        """
        # Add validation rules here
        # For example, check if controller parameters match the control mode
        if sim_params["control_mode"] == "constant":
            return "constant_torque" in sim_params
        elif sim_params["control_mode"] == "pd":
            return all(k in sim_params for k in ["target_angle", "kp", "kd"])
        return True
    
    def _save_sweep_results(self, results: List[Dict[str, Any]]):
        """Save sweep results to a JSON file.
        
        Args:
            results (List[Dict[str, Any]]): Sweep results
        """
        output_file = os.path.join(self.base_dir, "4_data/parameter_sweep_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Sweep results saved to {output_file}")

def main():
    # Create parameter sweep instance
    sweep = ParameterSweep()
    
    # Example: Run sweep with specific parameter combinations
    xml_combinations = [
        {"n_links": 1, "link_length": 0.15, "link_radius": 0.01, 
         "link_density": 1000.0, "joint_damping": 0.1, "joint_friction": 0.0},
        {"n_links": 2, "link_length": 0.15, "link_radius": 0.01,
         "link_density": 1000.0, "joint_damping": 0.1, "joint_friction": 0.0}
    ]
    
    sim_combinations = [
        {"control_mode": "pd", "initial_angle": 0.0, "sim_time": 10.0,
         "target_angle": 45.0, "kp": 3.0, "kd": 0.01},
        {"control_mode": "constant", "initial_angle": 0.0, "sim_time": 10.0,
         "constant_torque": 0.5}
    ]
    
    # Run sweep
    sweep.run_sweep(xml_combinations, sim_combinations)

if __name__ == "__main__":
    main() 