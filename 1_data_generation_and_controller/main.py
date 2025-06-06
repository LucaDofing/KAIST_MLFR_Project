import os
import json
import itertools
import subprocess
import argparse
from typing import Dict, List, Any, Union
import numpy as np
from sweep_config import robot_model_params, simulation_sweep_params

class ParameterSweep:
    def __init__(self, render: bool = False, base_dir: str = None):
        """Initialize the parameter sweep.
        Args:
            render (bool): Whether to render the simulation
            base_dir (str): Base directory for the project. If None, uses the directory of this script.
        """
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
        
        self.xml_dir = os.path.join(self.base_dir, "4_data/1_xml_models")
        self.render = render
        
        # Create directories if they don't exist
        os.makedirs(self.xml_dir, exist_ok=True)
        
        # Use imported parameter configurations
        self.robot_model = robot_model_params
        self.sim_sweep_params = simulation_sweep_params
    
    def _create_robot_folder_name(self) -> str:
        """Create a descriptive folder name based on robot parameters."""
        params = self.robot_model
        
        # Include fingertip mass in folder name if specified
        fingertip_suffix = ""
        if params.get('fingertip_mass') is not None:
            fingertip_suffix = f"_ftip{params['fingertip_mass']:.3f}"
        
        folder_name = (f"robot_L{params['n_links']}_"
                      f"len{params['link_length']:.2f}_"
                      f"rad{params['link_radius']:.3f}_"
                      f"mass{params['link_mass']:.1f}{fingertip_suffix}_"
                      f"damp{params['joint_damping']:.2f}_"
                      f"torq{params['torque_limit']:.1f}")
        return folder_name
    
    def generate_xml_for_robot(self) -> str:
        """Generate XML file for the robot model."""
        # Create XML filename based on robot parameters
        xml_filename = f"{self._create_robot_folder_name()}.xml"
        xml_path = os.path.join(self.xml_dir, xml_filename)
        
        # Generate XML using link_mass instead of density
        cmd = [
            "python3", os.path.join(self.base_dir, "1_xml_generator/1_generate_n_link_robot_xml.py"),
            "--num_links", str(self.robot_model["n_links"]),
            "--link_length", str(self.robot_model["link_length"]),
            "--link_radius", str(self.robot_model["link_radius"]),
            "--link_mass", str(self.robot_model["link_mass"]),  # Using mass instead of density
            "--joint_damping", str(self.robot_model["joint_damping"]),
            "--torque_limit", str(self.robot_model["torque_limit"]),  # Add torque limit
            "--output_dir", self.xml_dir
        ]
        
        # Add fingertip_mass parameter if specified
        if self.robot_model.get("fingertip_mass") is not None:
            cmd.extend(["--fingertip_mass", str(self.robot_model["fingertip_mass"])])
            print(f"Using specified fingertip mass: {self.robot_model['fingertip_mass']:.6f} kg")
        else:
            print(f"Using auto-calculated fingertip mass (from link density)")
        
        print(f"Generating XML for robot: {xml_filename}")
        subprocess.run(cmd, check=True)
        
        # Rename the default output to our specific filename
        default_xml = os.path.join(self.xml_dir, "n_link_robot.xml")
        if os.path.exists(default_xml):
            os.rename(default_xml, xml_path)
        
        return xml_path
    
    def run_simulation(self, xml_path: str, sim_params: Dict[str, Any], robot_data_dir: str) -> str:
        """Run a single simulation with given parameters."""
        
        # Get robot folder name
        robot_folder_name = self._create_robot_folder_name()
        
        cmd = [
            "python3", os.path.join(self.base_dir, "2_mujoco_sim/n_link_robot_mujoco.py"),
            "--xml_path", xml_path,
            "--sim_time", str(sim_params["sim_time"]),
            "--control_mode", sim_params["control_mode"],
            "--initial_angle", str(sim_params["initial_angle"]),
            "--log", "1",
            "--robot_folder_name", robot_folder_name
        ]
        
        if not self.render:
            cmd.append("--no-render")
        
        # Add controller-specific parameters
        if sim_params["control_mode"] == "constant":
            cmd.extend(["--constant_torque", str(sim_params["constant_torque"])])
        elif sim_params["control_mode"] == "pd":
            cmd.extend([
                "--target_angle", str(sim_params["target_angle"]),
                "--kp", str(sim_params["kp"]),
                "--kd", str(sim_params["kd"])
            ])
        
        print(f"Running simulation: init={sim_params['initial_angle']:.1f}°, "
              f"target={sim_params['target_angle']:.1f}°, "
              f"kp={sim_params['kp']}, kd={sim_params['kd']}")
        
        subprocess.run(cmd, check=True)
        
        # Find the most recently created data file
        mujoco_data_dir = os.path.join(self.base_dir, "4_data/2_mujoco")
        data_files = [f for f in os.listdir(mujoco_data_dir) if f.endswith(".json")]
        if not data_files:
            raise RuntimeError("No data file was generated")
        
        latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(mujoco_data_dir, x)))
        
        # Move the file to the robot-specific directory
        src_path = os.path.join(mujoco_data_dir, latest_file)
        dst_path = os.path.join(robot_data_dir, latest_file)
        os.rename(src_path, dst_path)
        
        return dst_path
    
    def run_sweep(self):
        """Run the complete parameter sweep for the defined robot model."""
        # Create robot-specific data directory
        robot_folder_name = self._create_robot_folder_name()
        robot_data_dir = os.path.join(self.base_dir, "4_data/2_mujoco/datasets", robot_folder_name)
        os.makedirs(robot_data_dir, exist_ok=True)
        
        print(f"Creating dataset for robot: {robot_folder_name}")
        print(f"Data will be saved to: {robot_data_dir}")
        
        # Show robot configuration including fingertip mass
        print(f"\nRobot configuration:")
        for key, value in self.robot_model.items():
            print(f"  {key}: {value}")
        
        # Generate XML for the robot model (only once)
        xml_path = self.generate_xml_for_robot()
        
        # Generate all simulation parameter combinations
        sim_combinations = self._generate_param_combinations(self.sim_sweep_params)
        
        print(f"Running {len(sim_combinations)} simulations...")
        
        results = []
        for i, sim_params in enumerate(sim_combinations, 1):
            print(f"\nSimulation {i}/{len(sim_combinations)}")
            
            # Skip invalid combinations
            if not self._is_valid_combination(sim_params):
                continue
            
            try:
                data_path = self.run_simulation(xml_path, sim_params, robot_data_dir)
                results.append({
                    "robot_model": self.robot_model,
                    "sim_params": sim_params,
                    "xml_path": xml_path,
                    "data_path": data_path,
                    "simulation_id": i
                })
            except Exception as e:
                print(f"Error in simulation {i}: {e}")
                continue
        
        # Save sweep metadata
        self._save_sweep_metadata(results, robot_data_dir)
        
        print(f"\nDataset generation complete!")
        print(f"Generated {len(results)} simulations")
        print(f"Robot model: {self.robot_model}")
        print(f"Data saved to: {robot_data_dir}")
    
    def _generate_param_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of simulation parameters."""
        keys = params.keys()
        values = params.values()
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        return combinations
    
    def _is_valid_combination(self, sim_params: Dict[str, Any]) -> bool:
        """Check if simulation parameter combination is valid."""
        if sim_params["control_mode"] == "constant":
            return "constant_torque" in sim_params
        elif sim_params["control_mode"] == "pd":
            return all(k in sim_params for k in ["target_angle", "kp", "kd"])
        return True
    
    def _save_sweep_metadata(self, results: List[Dict[str, Any]], robot_data_dir: str):
        """Save metadata about the parameter sweep."""
        metadata = {
            "robot_model": self.robot_model,
            "simulation_parameters": self.sim_sweep_params,
            "total_simulations": len(results),
            "successful_simulations": len([r for r in results if "data_path" in r]),
            "dataset_folder": robot_data_dir,
            "results": results
        }
        
        output_file = os.path.join(robot_data_dir, "dataset_metadata.json")
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Dataset metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for MuJoCo n-link robot with parameter sweep")
    parser.add_argument('--render', action='store_true', help='Enable rendering (visualization)')
    args = parser.parse_args()
    
    sweep = ParameterSweep(render=args.render)
    sweep.run_sweep()

if __name__ == "__main__":
    main() 