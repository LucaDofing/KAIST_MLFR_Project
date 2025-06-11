#!/usr/bin/env python3
"""
Dataset Generation Automation for N-Link Robot Simulation

This script automates the generation of large datasets by:
1. Loading robot model parameters from sweep_config.py
2. Automatically generating XML models using robot specifications
3. Running parameter sweeps across different simulation conditions
4. Organizing data into structured directories
5. Generating comprehensive datasets for machine learning applications

Author: KAIST MLFR Project
"""

import os
import sys
import json
import logging
import itertools
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sweep_config import robot_model_params, simulation_sweep_params
except ImportError as e:
    logger.error(f"Failed to import sweep_config: {e}")
    sys.exit(1)

# Import the XML models directory function for consistency
sys.path.append(os.path.join(os.path.dirname(__file__), "1_xml_generator"))

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "xml_gen", 
        os.path.join(os.path.dirname(__file__), "1_xml_generator", "1_generate_n_link_robot_xml.py")
    )
    if spec and spec.loader:
        xml_gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xml_gen)
    else:
        raise ImportError("Failed to load XML generator module")
except ImportError as e:
    logger.error(f"Failed to import XML generator: {e}")
    sys.exit(1)


class ParameterSweep:
    """
    Handles automated parameter sweep for n-link robot dataset generation.
    
    This class manages the complete workflow from robot model generation
    to parameter sweep execution and data organization.
    """
    
    def __init__(self, render: bool = False, base_dir: Optional[str] = None) -> None:
        """
        Initialize parameter sweep.
        
        Args:
            render: Whether to enable rendering during simulations
            base_dir: Base directory for the project. If None, uses the directory of this script
        """
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.render = render
        
        # Validate configuration parameters
        self._validate_config()
        
        # Use imported parameter configurations
        self.robot_model = robot_model_params.copy()
        self.sim_sweep_params = simulation_sweep_params.copy()
        
        logger.info("ParameterSweep initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_robot_params = ['n_links', 'link_length', 'link_radius', 'link_mass', 
                               'joint_damping', 'torque_limit']
        
        for param in required_robot_params:
            if param not in robot_model_params:
                raise ValueError(f"Missing required robot parameter: {param}")
        
        required_sim_params = ['control_mode', 'sim_time']
        for param in required_sim_params:
            if param not in simulation_sweep_params:
                raise ValueError(f"Missing required simulation parameter: {param}")
        
        logger.info("Configuration validation passed")
    
    def _create_robot_folder_name(self) -> str:
        """Create a descriptive folder name based on robot parameters."""
        params = self.robot_model
        
        # Include fingertip mass in folder name if specified
        fingertip_suffix = ""
        if params.get('fingertip_mass') is not None:
            fingertip_suffix = f"_ftip{params['fingertip_mass']:.3f}"
        
        folder_name = (
            f"robot_L{params['n_links']}_"
            f"len{params['link_length']:.2f}_"
            f"rad{params['link_radius']:.3f}_"
            f"mass{params['link_mass']:.1f}{fingertip_suffix}_"
            f"damp{params['joint_damping']:.2f}_"
            f"torq{params['torque_limit']:.1f}"
        )
        return folder_name
    
    def generate_xml_for_robot(self) -> str:
        """
        Generate XML file for the robot model.
        
        Returns:
            Path to the generated XML file
            
        Raises:
            subprocess.CalledProcessError: If XML generation fails
            FileNotFoundError: If generated XML file is not found
        """
        # Create XML filename based on robot parameters
        xml_filename = f"{self._create_robot_folder_name()}.xml"
        
        # Get the actual XML models directory (where files are really saved)
        actual_xml_dir = xml_gen.get_xml_models_dir()
        xml_path = os.path.join(actual_xml_dir, xml_filename)
        
        # Build command for XML generation
        cmd = [
            "python3", 
            str(self.base_dir / "1_xml_generator" / "1_generate_n_link_robot_xml.py"),
            "--num_links", str(self.robot_model["n_links"]),
            "--link_length", str(self.robot_model["link_length"]),
            "--link_radius", str(self.robot_model["link_radius"]),
            "--link_mass", str(self.robot_model["link_mass"]),
            "--joint_damping", str(self.robot_model["joint_damping"]),
            "--torque_limit", str(self.robot_model["torque_limit"]),
            "--output_name", xml_filename
        ]
        
        # Add fingertip_mass parameter if specified
        if self.robot_model.get("fingertip_mass") is not None:
            cmd.extend(["--fingertip_mass", str(self.robot_model["fingertip_mass"])])
            logger.info(f"Using specified fingertip mass: {self.robot_model['fingertip_mass']:.6f} kg")
        else:
            logger.info("Using auto-calculated fingertip mass (from link density)")
        
        logger.info(f"Generating XML for robot: {xml_filename}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"XML generation output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"XML generation failed: {e.stderr}")
            raise
        
        # Verify the file was created
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Generated XML file not found: {xml_path}")
        
        return xml_path
    
    def run_simulation(self, xml_path: str, sim_params: Dict[str, Any], robot_data_dir: str) -> str:
        """
        Run a single simulation with given parameters.
        
        Args:
            xml_path: Path to the robot XML model file
            sim_params: Simulation parameters dictionary
            robot_data_dir: Directory to save simulation data
            
        Returns:
            Path to the generated data file
            
        Raises:
            subprocess.CalledProcessError: If simulation fails
            RuntimeError: If no data file is generated
        """
        robot_folder_name = self._create_robot_folder_name()
        
        cmd = [
            "python3", 
            str(self.base_dir / "2_mujoco_sim" / "n_link_robot_mujoco.py"),
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
        
        logger.info(
            f"Running simulation: init={sim_params['initial_angle']:.1f}°, "
            f"target={sim_params.get('target_angle', 'N/A'):.1f}°, "
            f"kp={sim_params.get('kp', 'N/A')}, kd={sim_params.get('kd', 'N/A')}"
        )
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.debug(f"Simulation output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Simulation failed: {e.stderr}")
            raise
        
        # Find the most recently created data file
        mujoco_data_dir = self.base_dir / "4_data" / "2_mujoco"
        
        try:
            data_files = [f for f in os.listdir(mujoco_data_dir) if f.endswith(".json")]
            if not data_files:
                raise RuntimeError("No data file was generated")
            
            latest_file = max(
                data_files, 
                key=lambda x: os.path.getctime(mujoco_data_dir / x)
            )
            
            # Move the file to the robot-specific directory
            src_path = mujoco_data_dir / latest_file
            dst_path = Path(robot_data_dir) / latest_file
            
            os.rename(src_path, dst_path)
            
            return str(dst_path)
            
        except (OSError, IOError) as e:
            logger.error(f"Failed to handle data file: {e}")
            raise RuntimeError(f"Failed to handle data file: {e}")
    
    def run_sweep(self) -> None:
        """Run the complete parameter sweep for the defined robot model."""
        # Create robot-specific data directory
        robot_folder_name = self._create_robot_folder_name()
        robot_data_dir = self.base_dir / "4_data" / "2_mujoco" / "datasets" / robot_folder_name
        robot_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating dataset for robot: {robot_folder_name}")
        logger.info(f"Data will be saved to: {robot_data_dir}")
        
        # Show robot configuration including fingertip mass
        logger.info("Robot configuration:")
        for key, value in self.robot_model.items():
            logger.info(f"  {key}: {value}")
        
        # Generate XML for the robot model (only once)
        try:
            xml_path = self.generate_xml_for_robot()
        except Exception as e:
            logger.error(f"Failed to generate XML: {e}")
            return
        
        # Generate all simulation parameter combinations
        sim_combinations = self._generate_param_combinations(self.sim_sweep_params)
        
        logger.info(f"Running {len(sim_combinations)} simulations...")
        
        results = []
        successful_sims = 0
        
        for i, sim_params in enumerate(sim_combinations, 1):
            logger.info(f"Simulation {i}/{len(sim_combinations)}")
            
            # Skip invalid combinations
            if not self._is_valid_combination(sim_params):
                logger.warning(f"Skipping invalid parameter combination: {sim_params}")
                continue
            
            try:
                data_path = self.run_simulation(xml_path, sim_params, str(robot_data_dir))
                results.append({
                    "robot_model": self.robot_model,
                    "sim_params": sim_params,
                    "xml_path": xml_path,
                    "data_path": data_path,
                    "simulation_id": i
                })
                successful_sims += 1
                
            except Exception as e:
                logger.error(f"Error in simulation {i}: {e}")
                continue
        
        # Save sweep metadata
        try:
            self._save_sweep_metadata(results, str(robot_data_dir))
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
        
        logger.info("Dataset generation complete!")
        logger.info(f"Successful simulations: {successful_sims}/{len(sim_combinations)}")
        logger.info(f"Robot model: {self.robot_model}")
        logger.info(f"Data saved to: {robot_data_dir}")
    
    def _generate_param_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of simulation parameters."""
        if not params:
            return []
        
        keys = list(params.keys())
        values = list(params.values())
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _is_valid_combination(self, sim_params: Dict[str, Any]) -> bool:
        """Check if simulation parameter combination is valid."""
        control_mode = sim_params.get("control_mode")
        
        if control_mode == "constant":
            return "constant_torque" in sim_params
        elif control_mode == "pd":
            return all(k in sim_params for k in ["target_angle", "kp", "kd"])
        elif control_mode == "random":
            return True
        else:
            logger.warning(f"Unknown control mode: {control_mode}")
            return False
    
    def _save_sweep_metadata(self, results: List[Dict[str, Any]], robot_data_dir: str) -> None:
        """Save metadata about the parameter sweep."""
        metadata = {
            "robot_model": self.robot_model,
            "simulation_parameters": self.sim_sweep_params,
            "total_simulations": len(results),
            "successful_simulations": len([r for r in results if "data_path" in r]),
            "dataset_folder": robot_data_dir,
            "results": results
        }
        
        output_file = Path(robot_data_dir) / "dataset_metadata.json"
        
        try:
            with open(output_file, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Dataset metadata saved to {output_file}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save metadata: {e}")
            raise


def main() -> None:
    """Main entry point for the dataset generation script."""
    parser = argparse.ArgumentParser(
        description="Generate dataset for MuJoCo n-link robot with parameter sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--render', 
        action='store_true', 
        help='Enable rendering (visualization) - slower but useful for debugging'
    )
    parser.add_argument(
        '--no-render', 
        dest='render', 
        action='store_false', 
        help='Disable rendering for faster execution (default)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory for the project (default: script directory)'
    )
    parser.set_defaults(render=False)
    
    args = parser.parse_args()
    
    try:
        sweep = ParameterSweep(render=args.render, base_dir=args.base_dir)
        sweep.run_sweep()
    except KeyboardInterrupt:
        logger.info("Dataset generation interrupted by user")
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 