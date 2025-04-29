# N-Link Robot Simulation

A MuJoCo-based simulation of an n-link robot with configurable number of links.

## Project Structure

- `0_env_setup/`: Python environment setup
- `1_xml_generator/1_generate_n_link_robot_xml.py`: Script to generate the n-link robot XML model
- `2_mujoco_sim/n_link_robot_mujoco.py`: MuJoCo simulation for visualization and testing
- `3_gymnasium_sim/gym_minimal/n_link_robot.py`: Gymnasium environment for reinforcement learning
- `4_data/`: Data storage directory
  - `1_xml_models/`: Generated XML model files
  - `2_mujoco/`: MuJoCo simulation data
  - `3_gymnasium/`: Gymnasium simulation data

## Quick Start

### 1. Set up Environment

```bash
# Create and activate virtual environment
python3 -m venv 2dlinksim
source 2dlinksim/bin/activate

# Install dependencies 
pip install -r requirements.txt
```

### 2. Run the Simulation

```bash
# Generate an n-link robot model
python3 1_xml_generator/1_generate_n_link_robot_xml.py --num_links 2

# Run MuJoCo simulation 
python3 2_mujoco_sim/n_link_robot_mujoco.py

# Run with different control modes
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode pd
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode random
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode constant

# Run Gymnasium environment
python3 3_gymnasium_sim/gym_minimal/n_link_robot.py
```

## Data Storage

All simulation data is automatically saved in the `4_data` directory:

- XML models are generated in `4_data/1_xml_models/`
- MuJoCo simulation data is saved in `4_data/2_mujoco/`
- Gymnasium simulation data is saved in `4_data/3_gymnasium/`

## Features

- Configurable number of robot links
- Multiple control modes (PD, random, constant)
- Real-time visualization
- Data logging for analysis
- Gymnasium integration for reinforcement learning
