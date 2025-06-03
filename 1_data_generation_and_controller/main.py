import os
import json
import itertools
import subprocess
import argparse
from typing import Dict, List, Any, Union
import numpy as np
from sweep_config import xml_params, sim_params

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
        self.data_dir = os.path.join(self.base_dir, "4_data/2_mujoco")
        self.render = render
        # Create directories if they don't exist
        os.makedirs(self.xml_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        # Use imported parameter ranges
        self.xml_params = xml_params
        self.sim_params = sim_params
    def generate_xml(self, params: Dict[str, Any]) -> str:
        filename = f"n_link_robot_{'_'.join(f'{k}_{v}' for k, v in params.items())}.xml"
        xml_path = os.path.join(self.xml_dir, filename)
        cmd = [
            "python3", os.path.join(self.base_dir, "1_xml_generator/1_generate_n_link_robot_xml.py"),
            "--num_links", str(params["n_links"]),
            "--link_length", str(params["link_length"]),
            "--link_radius", str(params["link_radius"]),
            "--link_density", str(params["link_density"]),
            "--joint_damping", str(params["joint_damping"]),
            "--joint_friction", str(params["joint_friction"]),
            "--output_dir", self.xml_dir
        ]
        subprocess.run(cmd, check=True)
        default_xml = os.path.join(self.xml_dir, "n_link_robot.xml")
        if os.path.exists(default_xml):
            os.rename(default_xml, xml_path)
        return xml_path
    def run_simulation(self, xml_path: str, params: Dict[str, Any]) -> str:
        cmd = [
            "python3", os.path.join(self.base_dir, "2_mujoco_sim/n_link_robot_mujoco.py"),
            "--xml_path", xml_path,
            "--sim_time", str(params["sim_time"]),
            "--control_mode", params["control_mode"],
            "--initial_angle", str(params["initial_angle"]),
            "--log", "1"
        ]
        if not self.render:
            cmd.append("--no-render")
        # Add controller-specific parameters
        if params["control_mode"] == "constant":
            cmd.extend(["--constant_torque", str(params["constant_torque"])])
        elif params["control_mode"] == "pd":
            cmd.extend([
                "--target_angle", str(params["target_angle"]),
                "--kp", str(params["kp"]),
                "--kd", str(params["kd"])
            ])
        subprocess.run(cmd, check=True)
        data_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        if not data_files:
            raise RuntimeError("No data file was generated")
        latest_file = max(data_files, key=lambda x: os.path.getctime(os.path.join(self.data_dir, x)))
        return os.path.join(self.data_dir, latest_file)
    def run_sweep(self, xml_param_combinations: List[Dict[str, Any]] = None,
                 sim_param_combinations: List[Dict[str, Any]] = None):
        if xml_param_combinations is None:
            xml_param_combinations = self._generate_param_combinations(self.xml_params)
        if sim_param_combinations is None:
            sim_param_combinations = self._generate_param_combinations(self.sim_params)
        results = []
        for xml_params in xml_param_combinations:
            xml_path = self.generate_xml(xml_params)
            for sim_params in sim_param_combinations:
                if not self._is_valid_combination(xml_params, sim_params):
                    continue
                data_path = self.run_simulation(xml_path, sim_params)
                results.append({
                    "xml_params": xml_params,
                    "sim_params": sim_params,
                    "xml_path": xml_path,
                    "data_path": data_path
                })
        self._save_sweep_results(results)
    def _generate_param_combinations(self, params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = params.keys()
        values = params.values()
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        return combinations
    def _is_valid_combination(self, xml_params: Dict[str, Any], sim_params: Dict[str, Any]) -> bool:
        if sim_params["control_mode"] == "constant":
            return "constant_torque" in sim_params
        elif sim_params["control_mode"] == "pd":
            return all(k in sim_params for k in ["target_angle", "kp", "kd"])
        return True
    def _save_sweep_results(self, results: List[Dict[str, Any]]):
        output_file = os.path.join(self.base_dir, "4_data/parameter_sweep_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Sweep results saved to {output_file}")
def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for MuJoCo n-link robot")
    parser.add_argument('--render', action='store_true', help='Enable rendering (visualization)')
    args = parser.parse_args()
    sweep = ParameterSweep(render=args.render)
    sweep.run_sweep()
if __name__ == "__main__":
    main()
