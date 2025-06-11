# N-Link Robot Simulation & Dataset Generation

A comprehensive MuJoCo-based simulation framework for n-link robots with configurable physics, multiple control modes, and automated dataset generation capabilities.

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Navigate to the project directory
cd KAIST_MLFR_Project/1_data_generation_and_controller

# Create and activate virtual environment
python3 -m venv 0_2dlinksim
source 0_2dlinksim/bin/activate  # On Windows: 2dlinksim\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

**Required Python packages:**
- `mujoco` - Physics simulation engine
- `matplotlib` - Data visualization and plotting
- `numpy` - Numerical computations
- `glfw` - OpenGL window management for visualization

---

## 🔧 XML Model Generation

### Configuration System

The XML generator uses a **three-tier configuration system**:

1. **Default Parameters** (`1_xml_generator/parameters.py`) - Base configuration
2. **Command Line Arguments** - Override specific parameters
3. **main.py Parameter Sweeps** - Automatic override for dataset generation

### Understanding parameters.py - Base Configuration Layer

The `1_xml_generator/parameters.py` file serves as the **foundation** of all robot configurations. It defines default values that are used when no other input is provided, but **simulations can still be modified with custom user inputs**.

#### **View Current Default Configuration**

```bash
# Display all current default parameters
python3 1_xml_generator/parameters.py
```

**Example Output:**
```
🔧 Robot Structure:
  n_links: 1
  link_length: 0.15
  link_radius: 0.5
  link_mass: 100
  fingertip_mass: None

⚙️  Joint Parameters:
  joint_damping: 0.1
  joint_friction: 0.0
  joint_range: (-200.0, 200.0)

🔌 Actuator Parameters:
  torque_limit: 0.5
  motor_gear: 1.0

🌍 Physics Parameters:
  timestep: 0.01
  integrator: implicit
  gravity: (0, 0, -9.81)
```

#### **How Parameter Override Works**

The system follows this **hierarchy** (lowest to highest priority):

```
parameters.py defaults 
    ↓ (overridden by)
Command line --flags 
    ↓ (overridden by)  
main.py sweep_config.py
```

**Example Scenarios:**

```bash
# Uses ALL defaults from parameters.py
python3 1_xml_generator/1_generate_n_link_robot_xml.py
# Result: 1-link robot, 100kg mass, 0.5 N⋅m torque limit

# Override specific parameters with custom user inputs
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
    --num_links 3 \
    --torque_limit 25.0
# Result: 3-link robot, 100kg mass (from parameters.py), 25.0 N⋅m torque limit

# main.py uses sweep_config.py settings (overrides everything)
python3 main.py --no-render
# Result: Uses robot_model_params from sweep_config.py
```

#### **Customizing Defaults**

Edit `1_xml_generator/parameters.py` to change system-wide defaults, then use custom user inputs via command line flags to further modify as needed.

### View Current Configuration

```bash


### Basic XML Generation

Generate robot models using the XML generator script:

```bash
# Generate with all default parameters
python3 1_xml_generator/1_generate_n_link_robot_xml.py

# Generate with custom parameters
python3 1_xml_generator/1_generate_n_link_robot_xml.py [OPTIONS]
```

### Complete XML Generation Options

#### **Robot Structure Parameters**
```bash
--num_links N             # Number of links in the robot (default: 1)
--link_length FLOAT       # Length of each link in meters (default: 0.15)
--link_radius FLOAT       # Radius of each link in meters (default: 0.01)
--link_mass FLOAT         # Mass of each link in kg (default: 1.5)
--fingertip_mass FLOAT    # Mass of fingertip/gripper in kg (default: auto-calculated)
```

#### **Joint Parameters**
```bash
--joint_damping FLOAT     # Joint damping coefficient (default: 0.1)
--joint_friction FLOAT    # Joint friction coefficient (default: 0.0)
--joint_armature FLOAT    # Joint armature inertia (default: 0.0)
--joint_stiffness FLOAT   # Joint stiffness (default: 0.0)
--joint_springref FLOAT   # Joint spring reference position (default: 0.0)
--joint_range FLOAT FLOAT # Joint range in degrees (default: -200 200)
```

#### **Actuator Parameters**
```bash
--torque_limit FLOAT      # Maximum torque in N⋅m (default: 25.0)
--motor_gear FLOAT        # Motor gear ratio (default: 1.0)
```

#### **Physics Parameters**
```bash
--timestep FLOAT          # Simulation timestep in seconds (default: 0.01)
--integrator STR          # Integration method: implicit|explicit (default: implicit)
--gravity FLOAT FLOAT FLOAT # Gravity vector (default: 0 0 -9.81)
```

#### **Output Options**
```bash
--output_dir PATH         # Output directory for XML file (default: current directory)
--output_name STR         # Custom filename for XML (default: n_link_robot.xml)
```

### XML Generation Examples

```bash
# Generate a simple 2-link robot
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
  --num_links 2 \
  --output_dir 4_data/1_xml_models

# Generate a heavy-duty single-link robot
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
  --num_links 1 \
  --link_length 0.5 \
  --link_mass 10.0 \
  --torque_limit 50.0 \
  --joint_damping 0.5

# Generate a precise manipulation robot
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
  --num_links 3 \
  --link_length 0.2 \
  --link_radius 0.02 \
  --fingertip_mass 0.1 \
  --joint_damping 0.01 \
  --timestep 0.001
```

---

## 🎮 Single Simulation Execution

### Basic Simulation

Run individual simulations using the MuJoCo simulation script:

```bash
python3 2_mujoco_sim/n_link_robot_mujoco.py [OPTIONS]
```

### Complete Simulation Options

#### **Model & Output**
```bash
--xml_path PATH           # Path to MuJoCo XML file (default: 4_data/1_xml_models/n_link_robot.xml)
--log BOOL                # Enable data logging: 0|1 (default: 0)
--robot_folder_name STR   # Custom folder name for organizing data
```

#### **Control Modes & Parameters**

**1. Constant Torque Control**
```bash
--control_mode constant
--constant_torque FLOAT   # Constant torque value in N⋅m (default: 0.5)
```

**2. PD (Proportional-Derivative) Control**
```bash
--control_mode pd
--target_angle FLOAT      # Target angle in degrees (default: 20.0)
--kp FLOAT               # Proportional gain (default: 3.0)
--kd FLOAT               # Derivative gain (default: 0.010)
```

**3. Random Control**
```bash
--control_mode random     # Random torque control
```

#### **Initial Conditions**
```bash
--initial_angle FLOAT     # Initial angle in degrees for all joints (default: 0.0)
```

#### **Simulation Parameters**
```bash
--sim_time FLOAT          # Simulation duration in seconds (default: 20.0)
--speedup FLOAT           # Simulation speedup factor (default: 1.0)
```

#### **Visualization Options**
```bash
--render                  # Enable real-time visualization (default)
--no-render              # Disable visualization for faster execution
```

### Single Simulation Examples

```bash
# Visualized PD control simulation
python3 2_mujoco_sim/n_link_robot_mujoco.py \
  --xml_path 4_data/1_xml_models/n_link_robot.xml \
  --control_mode pd \
  --target_angle 45.0 \
  --kp 10.0 \
  --kd 0.1 \
  --initial_angle 0.0 \
  --sim_time 5.0 \
  --log 1

# Fast constant torque simulation without visualization
python3 2_mujoco_sim/n_link_robot_mujoco.py \
  --control_mode constant \
  --constant_torque 2.0 \
  --sim_time 10.0 \
  --no-render \
  --log 1

# Random control with specific initial conditions
python3 2_mujoco_sim/n_link_robot_mujoco.py \
  --control_mode random \
  --initial_angle 30.0 \
  --sim_time 15.0 \
  --speedup 2.0
```

---

## 📊 Dataset Generation with main.py

### Overview

The `main.py` script automates the generation of large datasets by:
1. **Loading default XML parameters** from `1_xml_generator/parameters.py`
2. **Automatically generating XML models** using robot specifications from `sweep_config.py`
3. **Running parameter sweeps** across different simulation conditions
4. **Organizing data** into structured directories
5. **Generating comprehensive datasets** for machine learning applications

### Configuration Hierarchy

The system uses a **hierarchical configuration approach**:

```
1. Base Defaults (parameters.py) 
   ↓ 
2. Robot Model (sweep_config.py) 
   ↓ 
3. Simulation Sweeps (sweep_config.py)
```

**Example:** If `parameters.py` sets `link_mass: 1.5` but `sweep_config.py` sets `link_mass: 5.0`, the final robot will use `5.0 kg`.

### Configuration Setup

Edit `sweep_config.py` to define your dataset parameters:

```python
# Robot model parameters - define the physical robot
robot_model_params = {
    "n_links": 2,                    # Number of links
    "link_length": 0.30,             # Link length in meters
    "link_radius": 0.05,             # Link radius in meters
    "link_mass": 5.0,                # Link mass in kg
    "fingertip_mass": 0.2,           # Fingertip mass in kg (None = auto-calculate)
    "joint_damping": 0.5,            # Joint damping coefficient
    "torque_limit": 25.0,            # Maximum torque in N⋅m
}

# Simulation parameters - vary these to create the dataset
simulation_sweep_params = {
    "control_mode": ["pd", "constant"],           # Control strategies
    "initial_angle": [0.0, 15.0, 30.0],         # Initial positions (degrees)
    "target_angle": [45.0, 90.0, 135.0],        # Target positions (degrees)
    "kp": [5.0, 10.0, 20.0],                    # Proportional gains
    "kd": [0.01, 0.05, 0.1],                    # Derivative gains
    "constant_torque": [1.0, 2.0, 3.0],         # Constant torque values
    "sim_time": [4.0, 6.0],                     # Simulation durations
}
```

### Dataset Generation Commands

```bash
# Generate dataset with visualization (slower)
python3 main.py --render

# Generate dataset without visualization (faster - recommended)
python3 main.py --no-render

```

### Dataset Structure

Generated datasets are organized as follows:

```
4_data/2_mujoco/datasets/
└── robot_L2_len0.30_rad0.050_mass5.0_ftip0.200_damp0.50_torq25.0/
    ├── sweep_metadata.json                    # Dataset overview
    ├── trajectory_20250108_143052_...json     # Individual simulation data
    ├── trajectory_20250108_143053_...json
    └── ...
```

### Understanding Dataset Scale

**Parameter Combinations:** The dataset size equals the product of all parameter list lengths.

**Example calculation:**
- `control_mode`: 2 options (pd, constant)
- `initial_angle`: 3 options (0°, 15°, 30°)
- `target_angle`: 3 options (45°, 90°, 135°)
- `kp`: 3 options (5.0, 10.0, 20.0)
- `kd`: 3 options (0.01, 0.05, 0.1)
- `constant_torque`: 3 options (1.0, 2.0, 3.0)
- `sim_time`: 2 options (4.0s, 6.0s)

**Total simulations:** 2 × 3 × 3 × 3 × 3 × 3 × 2 = **972 simulations**

### Large Dataset Generation Strategies

**1. Moderate Dataset (100-500 simulations)**
```python
simulation_sweep_params = {
    "control_mode": ["pd"],
    "initial_angle": [0.0, 30.0, 60.0],         # 3 options
    "target_angle": [45.0, 90.0],               # 2 options  
    "kp": [5.0, 15.0, 25.0],                    # 3 options
    "kd": [0.01, 0.1],                          # 2 options
    "sim_time": [5.0],                          # 1 option
}
# Total: 3 × 2 × 3 × 2 × 1 = 36 simulations
```

**2. Large Dataset (1000+ simulations)**
```python
simulation_sweep_params = {
    "control_mode": ["pd", "constant"],
    "initial_angle": list(range(0, 91, 15)),     # [0, 15, 30, 45, 60, 75, 90] = 7 options
    "target_angle": list(range(45, 181, 15)),    # [45, 60, ..., 180] = 10 options
    "kp": [1.0, 5.0, 10.0, 20.0, 50.0],        # 5 options
    "kd": [0.005, 0.01, 0.05, 0.1, 0.2],       # 5 options
    "constant_torque": [0.5, 1.0, 2.0, 3.0],    # 4 options
    "sim_time": [3.0, 5.0, 8.0],               # 3 options
}
# Total: 2 × 7 × 10 × 5 × 5 × 4 × 3 = 21,000 simulations
```


## 📈 Data Analysis & Visualization

### Plotting Generated Data

Use the plotting script to visualize simulation results:

```bash
cd 5_plots

# Edit the filename in plot.py (line 8):
filename = "trajectory_20250608_104447_n_link_2_init_0.0_target_20.0_kp_3.0_kd_0.010_damping_0.100.json"

# Run the plotting script
python3 plot.py
```

The plot will display:
- **Theta**: Joint angles over time
- **Omega**: Joint velocities over time  
- **Alpha**: Joint accelerations over time
- **Torque**: Applied torques over time

---

## 🔧 Advanced Usage

### Custom Robot Configurations

**Heavy Industrial Robot:**
```bash
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
  --num_links 1 \
  --link_length 1.0 \
  --link_mass 50.0 \
  --torque_limit 200.0 \
  --joint_damping 2.0
```

**Precision Manipulation Robot:**
```bash
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
  --num_links 3 \
  --link_length 0.1 \
  --link_mass 0.5 \
  --torque_limit 5.0 \
  --joint_damping 0.01 \
  --timestep 0.001
```

### Performance Optimization

**Fast Dataset Generation:**
```bash
# Minimize simulation time and disable rendering
python3 main.py --no-render

# Use shorter simulation times in sweep_config.py
"sim_time": [2.0]  # Instead of [5.0, 10.0]
```

**High-Fidelity Simulations:**
```bash
# Use smaller timesteps and longer simulations
# In XML generation:
--timestep 0.001

# In sweep_config.py:
"sim_time": [10.0, 15.0]
```

---

## 📁 Project Structure

```
KAIST_MLFR_Project/1_data_generation_and_controller/
├── 0_env_setup/              # Environment setup scripts
├── 1_xml_generator/           # Robot model generation
│   └── 1_generate_n_link_robot_xml.py
├── 2_mujoco_sim/             # MuJoCo simulation engine
│   ├── n_link_robot_mujoco.py
│   ├── controllers.py
│   └── data_logger.py
├── 3_gymnasium_sim/          # Gymnasium RL environment
├── 4_data/                   # Data storage
│   ├── 1_xml_models/         # Generated robot models
│   ├── 2_mujoco/            # Simulation data
│   └── 3_gymnasium/         # RL training data
├── 5_plots/                  # Visualization scripts
│   └── plot.py
├── main.py                   # Dataset generation automation
├── sweep_config.py          # Parameter sweep configuration
└── README.md               # This documentation
```

---

## 🔄 Quick Examples

**Explore the configuration system:**
```bash
# 1. View default configuration
python3 1_xml_generator/parameters.py

# 2. Generate robot with defaults
python3 1_xml_generator/1_generate_n_link_robot_xml.py

# 3. Override specific parameters
python3 1_xml_generator/1_generate_n_link_robot_xml.py --num_links 2 --link_mass 3.0

# 4. Generate full dataset (uses sweep_config.py overrides)
python3 main.py --no-render

# 5. Visualize results
cd 5_plots && python3 plot.py
```

**Configuration flexibility examples:**
```bash
# Modify default config (edit parameters.py), then:
python3 1_xml_generator/1_generate_n_link_robot_xml.py  # Uses your new defaults

# Override defaults with command line:
python3 1_xml_generator/1_generate_n_link_robot_xml.py --torque_limit 100.0

# Override everything with sweep_config.py:
python3 main.py --no-render  # Uses sweep_config.py robot parameters
```

This comprehensive framework enables rapid prototyping, systematic dataset generation, and detailed analysis of n-link robot dynamics under various control strategies.
