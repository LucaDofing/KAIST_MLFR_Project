#!/usr/bin/env python3
"""
Parameter Sweep Configuration for N-Link Robot Dataset Generation

This file defines the robot model parameters and simulation sweep parameters
for automated dataset generation. The configuration uses a hierarchical approach:

1. Robot Model Parameters: Define the physical robot (one robot per sweep)
2. Simulation Sweep Parameters: Vary control conditions to create the dataset

Author: KAIST MLFR Project
"""

from typing import Dict, List, Union, Any

# =============================================================================
# ROBOT MODEL PARAMETERS
# =============================================================================
# These parameters define the physical robot model. The XML generator will
# create a single robot model based on these specifications.

robot_model_params: Dict[str, Any] = {
    # Robot structure
    "n_links": 1,                    # Number of links in the robot
    "link_length": 0.40,             # Length of each link in meters
    "link_radius": 0.075,            # Radius of each link in meters
    "link_mass": 7.0,                # Mass of each link in kg
    "fingertip_mass": 0.5,           # Mass of fingertip/gripper in kg (None = auto-calculate)
    
    # Joint properties
    "joint_damping": 0.7,            # Joint damping coefficient
    "joint_friction": 0.0,           # Joint friction coefficient (optional)
    "joint_range": (-200.0, 200.0),  # Joint range in degrees (optional)
    
    # Actuator properties
    "torque_limit": 20.0,            # Maximum torque in N⋅m
    "motor_gear": 1.0,               # Motor gear ratio (optional)
    
    # Physics settings (optional - will use defaults if not specified)
    "timestep": 0.01,                # Simulation timestep in seconds
    "integrator": "implicit",        # Integration method: implicit|explicit
}

# =============================================================================
# SIMULATION SWEEP PARAMETERS
# =============================================================================
# These parameters define the sweep across different simulation conditions.
# Each parameter list creates multiple simulation runs.
# Total simulations = product of all list lengths

simulation_sweep_params: Dict[str, List[Union[str, float, int]]] = {
    # Control strategy
    "control_mode": ["constant"],          # Options: "pd", "constant", "random"
    
    # Initial conditions
    "initial_angle": [0.0],          # Initial joint angles in degrees
    
    # PD controller parameters (used when control_mode includes "pd")
    "target_angle": [45.0],          # Target angles in degrees
    "kp": [30.0, 10.0],              # Proportional gains
    "kd": [0.010],                   # Derivative gains
    
    # Constant controller parameters (used when control_mode includes "constant")
    "constant_torque": [1.0, 2.0],   # Constant torque values in N⋅m
    
    # Simulation duration
    "sim_time": [4.0],               # Simulation time in seconds
}

# =============================================================================
# CONFIGURATION VALIDATION AND EXAMPLES
# =============================================================================

def validate_config() -> None:
    """Validate the configuration parameters."""
    # Validate robot model parameters
    required_robot_params = ['n_links', 'link_length', 'link_radius', 'link_mass', 
                           'joint_damping', 'torque_limit']
    
    for param in required_robot_params:
        if param not in robot_model_params:
            raise ValueError(f"Missing required robot parameter: {param}")
        if not isinstance(robot_model_params[param], (int, float)):
            raise ValueError(f"Robot parameter {param} must be numeric")
    
    # Validate simulation sweep parameters
    required_sim_params = ['control_mode', 'sim_time']
    
    for param in required_sim_params:
        if param not in simulation_sweep_params:
            raise ValueError(f"Missing required simulation parameter: {param}")
        if not isinstance(simulation_sweep_params[param], list):
            raise ValueError(f"Simulation parameter {param} must be a list")
    
    # Check for PD controller requirements
    if "pd" in simulation_sweep_params.get("control_mode", []):
        pd_params = ['target_angle', 'kp', 'kd']
        for param in pd_params:
            if param not in simulation_sweep_params:
                raise ValueError(f"PD control requires parameter: {param}")
    
    # Check for constant controller requirements
    if "constant" in simulation_sweep_params.get("control_mode", []):
        if "constant_torque" not in simulation_sweep_params:
            raise ValueError("Constant control requires parameter: constant_torque")


def get_total_simulations() -> int:
    """Calculate the total number of simulations."""
    total = 1
    for param_list in simulation_sweep_params.values():
        if isinstance(param_list, list):
            total *= len(param_list)
    return total


def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("DATASET GENERATION CONFIGURATION")
    print("=" * 60)
    
    print("\n🤖 Robot Model:")
    print(f"  Links: {robot_model_params['n_links']}")
    print(f"  Length: {robot_model_params['link_length']:.3f} m")
    print(f"  Radius: {robot_model_params['link_radius']:.3f} m")
    print(f"  Mass: {robot_model_params['link_mass']:.1f} kg")
    print(f"  Fingertip mass: {robot_model_params['fingertip_mass']} kg")
    print(f"  Joint damping: {robot_model_params['joint_damping']:.3f}")
    print(f"  Torque limit: {robot_model_params['torque_limit']:.1f} N⋅m")
    
    print("\n📊 Simulation Sweep:")
    for param, values in simulation_sweep_params.items():
        print(f"  {param}: {values} ({len(values)} options)")
    
    total_sims = get_total_simulations()
    print(f"\n📈 Total simulations: {total_sims}")
    
    # Estimate time
    est_time_per_sim = 3  # seconds (rough estimate)
    est_total_time = total_sims * est_time_per_sim
    print(f"⏱️  Estimated time: {est_total_time:.0f} seconds ({est_total_time/60:.1f} minutes)")
    
    print("=" * 60)


# =============================================================================
# ALTERNATIVE CONFIGURATION EXAMPLES
# =============================================================================

# Example 1: Large dataset for machine learning
LARGE_DATASET_CONFIG = {
    "control_mode": ["pd", "constant"],
    "initial_angle": [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0],  # 7 options
    "target_angle": [30.0, 45.0, 60.0, 90.0, 120.0, 150.0],      # 6 options
    "kp": [5.0, 10.0, 20.0, 30.0, 50.0],                         # 5 options
    "kd": [0.005, 0.01, 0.02, 0.05, 0.1],                        # 5 options
    "constant_torque": [0.5, 1.0, 2.0, 3.0, 5.0],                # 5 options
    "sim_time": [3.0, 5.0, 8.0],                                 # 3 options
    # Total: 2 × 7 × 6 × 5 × 5 × 5 × 3 = 15,750 simulations
}

# Example 2: Quick test dataset
QUICK_TEST_CONFIG = {
    "control_mode": ["pd"],
    "initial_angle": [0.0, 30.0],                                # 2 options
    "target_angle": [45.0, 90.0],                                # 2 options
    "kp": [10.0, 30.0],                                          # 2 options
    "kd": [0.01, 0.05],                                          # 2 options
    "sim_time": [3.0],                                           # 1 option
    # Total: 1 × 2 × 2 × 2 × 2 × 1 = 16 simulations
}

# Example 3: System identification dataset
SYSTEM_ID_CONFIG = {
    "control_mode": ["random", "constant"],
    "initial_angle": [0.0],                                      # 1 option
    "constant_torque": [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0], # 7 options
    "sim_time": [10.0, 15.0, 20.0],                             # 3 options
    # Total: 2 × 1 × 7 × 3 = 42 simulations
}


def use_alternative_config(config_name: str) -> None:
    """
    Switch to an alternative configuration.
    
    Args:
        config_name: Name of the configuration ("large", "quick_test", "system_id")
    """
    global simulation_sweep_params
    
    if config_name == "large":
        simulation_sweep_params = LARGE_DATASET_CONFIG.copy()
    elif config_name == "quick_test":
        simulation_sweep_params = QUICK_TEST_CONFIG.copy()
    elif config_name == "system_id":
        simulation_sweep_params = SYSTEM_ID_CONFIG.copy()
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    print(f"Switched to '{config_name}' configuration")


if __name__ == "__main__":
    # Run validation and print summary when script is executed directly
    try:
        validate_config()
        print("✅ Configuration validation passed")
        print_config_summary()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
