# N-Link Robot Simulation

A MuJoCo-based simulation of an n-link robot with configurable number of links and different control modes.

## Project Structure

- `0_env_setup/`: Python environment setup
- `1_xml_generator/1_generate_n_link_robot_xml.py`: Script to generate the n-link robot XML model
- `2_mujoco_sim/`: MuJoCo simulation
  - `n_link_robot_mujoco.py`: Main simulation script
  - `controllers.py`: Different control strategies
  - `data_logger.py`: Data logging functionality
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

### 2. Generate Robot Model

```bash
# Generate an n-link robot model (default: 2 links)
python3 1_xml_generator/1_generate_n_link_robot_xml.py --num_links 2
```

### 3. Run MuJoCo Simulation

The simulation can run in two modes: visualization mode and computation mode.

#### Visualization Mode (Default)
Shows real-time visualization and detailed state information:
```bash
# Run with visualization (default)
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode constant --constant_torque 3.0
```

#### Computation Mode
Runs without visualization for faster data generation:
```bash
# Run without visualization (faster)
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode constant --constant_torque 3.0 --no-render
```

### Control Modes

#### Constant Torque Control
```bash
# Run with constant torque (default: 5.0 Nm)
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode constant --constant_torque 3.0
```

#### PD (Proportional-Derivative) Control
```bash
# Run with PD control
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode pd --target_angle 45 --kp 100 --kd 10
```

#### Random Control
```bash
# Run with random torques
python3 2_mujoco_sim/n_link_robot_mujoco.py --control_mode random
```

### Command-line Options

- `--control_mode`: Control strategy to use (`constant`, `pd`, or `random`)
- `--constant_torque`: Torque value for constant control mode (default: 5.0 Nm)
- `--target_angle`: Target angle for PD control in degrees (default: 10.0°)
- `--kp`: Position gain for PD control (default: 100.0)
- `--kd`: Velocity gain for PD control (default: 10.0)
- `--sim_time`: Simulation duration in seconds (default: 2.0s)
- `--no-render`: Run without visualization for faster data generation

### Performance Tuning

The simulation performance can be adjusted through several parameters:

1. **Solver Timestep**: Set in the XML file (default: 0.01s)
   - Smaller timesteps = more accurate simulation but slower computation
   - Larger timesteps = faster but less accurate
   - All physics steps are logged to ensure complete data capture

2. **Rendering Frequency**: Control visualization speed
```bash
# Run at 60 FPS
python3 2_mujoco_sim/n_link_robot_mujoco.py --render_fps 60

# Run at 10 FPS for slow-motion visualization
python3 2_mujoco_sim/n_link_robot_mujoco.py --render_fps 10
```

3. **Computation Mode**: For fastest simulation
```bash
# Maximum speed without visualization
python3 2_mujoco_sim/n_link_robot_mujoco.py --no-render
```

### Data Logging

The simulation logs data at every physics step to ensure no dynamics information is lost. All simulation data is automatically saved in the `4_data` directory:
- XML models are generated in `4_data/1_xml_models/`
- MuJoCo simulation data (joint positions, velocities, torques) is saved in `4_data/2_mujoco/`
- Gymnasium simulation data is saved in `4_data/3_gymnasium/`

## Features

- Configurable number of robot links
- Multiple control modes:
  - Constant torque control
  - PD position control
  - Random torque control
- Two simulation modes:
  - Visualization mode with real-time display
  - Fast computation mode for data generation
- Comprehensive data logging
- Gymnasium integration for reinforcement learning
