# MuJoCo N-Link Robot Simulation & Dataset Generation

This project implements a comprehensive robotics simulation framework using MuJoCo for n-link robotic arms. It features XML robot model generation, physics simulation with multiple control modes, and automated dataset generation for machine learning applications.

---

## Project Structure

```
KAIST_MLFR_Project/
│
├── 1_data_generation_and_controller/
│   ├── parameters.py                    # 🎯 Base parameter defaults (Tier 1)
│   ├── main.py                         # Dataset generation with parameter sweeps
│   ├── sweep_config.py                 # Sweep parameter configurations
│   │
│   ├── 1_xml_generator/
│   │   └── 1_generate_n_link_robot_xml.py  # Robot XML model generator
│   │
│   └── 2_mujoco_sim/
│       ├── n_link_robot_mujoco.py      # MuJoCo simulation engine
│       ├── data_logger.py              # Trajectory data logging
│       └── plot.py                     # Data visualization
│
├── 4_data/
│   ├── 1_xml_models/                   # Generated robot XML files
│   └── 2_mujoco/
│       ├── datasets/                   # Organized dataset folders
│       │   └── robot_L3_len0.33_rad0.025_mass0.5_damp0.10_torq100.0/
│       │       ├── data/               # Trajectory JSON files
│       │       └── info/               # Metadata & XML models
│       └── *.json                      # Individual simulation files
│
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## Environment Setup

1. **Clone and navigate to the project:**
```bash
git clone <repository-url>
cd KAIST_MLFR_Project/1_data_generation_and_controller
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
- `mujoco` - Physics simulation engine
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `argparse` - Command line interface

---

## Three-Tier Configuration System

Our system uses a flexible three-tier parameter hierarchy:

### **Tier 1: Base Defaults** (`parameters.py`)
Centralized default parameters organized by categories:

```python
# View all default parameters
python3 parameters.py

# Example output:
Robot Structure Parameters:
  num_links: 3
  link_length: 0.33
  link_radius: 0.025
  link_mass: 0.5
  fingertip_mass: None (auto-calculated)

Joint Parameters:
  joint_damping: 0.1
  joint_friction: 0.0

Actuator Parameters:
  torque_limit: 100.0
```

### **Tier 2: Command Line Overrides**
Override any parameter via command line:

```bash
# Generate XML with custom parameters
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
    --num_links 5 \
    --link_mass 0.8 \
    --torque_limit 150.0 \
    --output_name custom_robot.xml

# View available parameters and their defaults
python3 1_xml_generator/1_generate_n_link_robot_xml.py --show_config
```

### **Tier 3: Parameter Sweeps** (`main.py`)
Automated dataset generation with parameter combinations defined in `sweep_config.py`:

```python
# Edit sweep_config.py to define parameter ranges
robot_model_params = {
    "n_links": 3,
    "link_length": 0.33,
    "link_mass": 0.5,
    "joint_damping": 0.1,
    "torque_limit": 100.0,
    "fingertip_mass": None  # Auto-calculated from link density
}

simulation_sweep_params = {
    "sim_time": [5.0],
    "control_mode": ["pd"],
    "initial_angle": np.linspace(-60, 60, 5),  # 5 initial positions
    "target_angle": np.linspace(-60, 60, 5),   # 5 target positions
    "kp": [10.0, 50.0],                       # 2 proportional gains
    "kd": [0.1, 0.5, 1.0]                     # 3 derivative gains
}
# Total combinations: 5 × 5 × 2 × 3 = 150 simulations
```

---

## XML Robot Model Generation

### **Basic Usage**
```bash
# Generate with default parameters
python3 1_xml_generator/1_generate_n_link_robot_xml.py

# Custom 5-link robot with heavy links
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
    --num_links 5 \
    --link_mass 1.0 \
    --link_length 0.4 \
    --torque_limit 200.0 \
    --output_name heavy_5link.xml
```

### **Parameter Categories**

#### **Robot Structure**
- `--num_links`: Number of links (default: 3)
- `--link_length`: Length of each link in meters (default: 0.33)
- `--link_radius`: Radius of cylindrical links (default: 0.025)
- `--link_mass`: Mass of each link in kg (default: 0.5)
- `--fingertip_mass`: Optional fingertip mass (default: auto-calculated)

#### **Joint Parameters**
- `--joint_damping`: Joint damping coefficient (default: 0.1)
- `--joint_friction`: Joint friction coefficient (default: 0.0)
- `--joint_stiffness`: Joint stiffness (default: 0.0)
- `--joint_range`: Joint angle limits in degrees (default: 180.0)

#### **Actuator Parameters**
- `--torque_limit`: Maximum actuator torque in N⋅m (default: 100.0)
- `--velocity_limit`: Maximum joint velocity (default: 10.0)
- `--force_limit`: Maximum actuator force (default: 1000.0)

#### **Physics Parameters**
- `--timestep`: Simulation timestep (default: 0.002)
- `--solver_iterations`: Solver iterations (default: 50)
- `--solver_tolerance`: Solver tolerance (default: 1e-10)

#### **Output Options**
- `--output_name`: Custom filename (default: auto-generated)
- `--output_dir`: Custom output directory (default: auto-detected)
- `--show_config`: Display current parameter configuration

### **Advanced Examples**
```bash
# High-precision simulation setup
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
    --timestep 0.001 \
    --solver_iterations 100 \
    --solver_tolerance 1e-12 \
    --output_name precision_robot.xml

# Heavy-duty industrial robot
python3 1_xml_generator/1_generate_n_link_robot_xml.py \
    --num_links 6 \
    --link_mass 2.0 \
    --link_length 0.5 \
    --torque_limit 500.0 \
    --joint_damping 0.5 \
    --output_name industrial_6link.xml
```

---

## Single Simulation Execution

### **Control Modes**

#### **1. Constant Torque**
Apply constant torque to all joints:
```bash
python3 2_mujoco_sim/n_link_robot_mujoco.py \
    --xml_path ../4_data/1_xml_models/robot_3link.xml \
    --control_mode constant \
    --constant_torque 5.0 \
    --sim_time 3.0 \
    --initial_angle 0.5
```

#### **2. PD Control**
Position control with proportional-derivative gains:
```bash
python3 2_mujoco_sim/n_link_robot_mujoco.py \
    --xml_path ../4_data/1_xml_models/robot_3link.xml \
    --control_mode pd \
    --initial_angle 45.0 \
    --target_angle -30.0 \
    --kp 10.0 \
    --kd 0.5 \
    --sim_time 5.0
```

#### **3. Random Control**
Random torque commands for exploration:
```bash
python3 2_mujoco_sim/n_link_robot_mujoco.py \
    --xml_path ../4_data/1_xml_models/robot_3link.xml \
    --control_mode random \
    --sim_time 10.0
```

### **Simulation Options**
- `--render` / `--no-render`: Enable/disable visualization (default: enabled)
- `--log`: Save trajectory data (0/1, default: 0)
- `--robot_folder_name`: Organize data in structured folders

---

## Dataset Generation with main.py

### **Quick Start**
```bash
# Generate dataset with current sweep_config.py settings
python3 main.py

# Generate dataset with visualization enabled
python3 main.py --render
```

### **Dataset Size Calculation**
The total number of simulations is the product of all parameter combinations:

**Example Configuration:**
```python
simulation_sweep_params = {
    "initial_angle": np.linspace(-60, 60, 6),    # 6 values
    "target_angle": np.linspace(-60, 60, 6),     # 6 values  
    "kp": [5.0, 10.0, 20.0, 50.0],             # 4 values
    "kd": [0.1, 0.5, 1.0]                      # 3 values
}
# Total: 6 × 6 × 4 × 3 = 432 simulations
```

### **Dataset Organization**
```
4_data/2_mujoco/datasets/robot_L3_len0.33_rad0.025_mass0.5_damp0.10_torq100.0/
├── data/                           # Individual trajectory files
│   ├── trajectory_20241208_143022_n_link_3_init_-60.0_target_-60.0_kp_5.0_kd_0.1_damping_0.100.json
│   ├── trajectory_20241208_143025_n_link_3_init_-60.0_target_-36.0_kp_5.0_kd_0.1_damping_0.100.json
│   └── ...
└── info/                           # Metadata and model files
    ├── dataset_metadata.json       # Dataset overview
    └── robot_model.xml             # Robot XML model
```

### **Performance Estimates**
- **Small Dataset** (36 sims): ~2-3 minutes
- **Medium Dataset** (432 sims): ~25-30 minutes  
- **Large Dataset** (2,160 sims): ~2-3 hours
- **Extra Large Dataset** (10,800 sims): ~8-12 hours

*Times assume 5-second simulations on modern hardware*

---

## Data Visualization

### **Plot Trajectory Data**
```bash
cd 2_mujoco_sim

# Plot a specific trajectory file (filename only needed)
python3 plot.py

# Then edit the filename at the top of plot.py:
filename = "trajectory_20241208_143022_n_link_3_init_-60.0_target_-36.0_kp_5.0_kd_0.1_damping_0.100.json"
```

**Generated Plots:**
- Joint angles over time
- Joint velocities over time  
- Joint accelerations over time
- Applied torques over time

---

## Advanced Usage

### **Custom Parameter Configurations**

1. **Edit Base Defaults** (`parameters.py`):
```python
# Modify default parameters for all operations
DEFAULT_PARAMS = {
    "robot_structure": {
        "num_links": 4,           # Change default to 4-link robot
        "link_mass": 0.8,         # Heavier links
        "torque_limit": 150.0,    # Higher torque capacity
    }
}
```

2. **Custom Sweep Configurations** (`sweep_config.py`):
```python
# Create focused parameter sweeps
robot_model_params = {
    "n_links": 3,
    "joint_damping": 0.2,        # Higher damping
    "fingertip_mass": 0.05,      # Explicit fingertip mass
}

simulation_sweep_params = {
    "control_mode": ["pd"],
    "initial_angle": [0.0],                    # Single initial position
    "target_angle": np.linspace(-90, 90, 19), # 19 target positions
    "kp": np.logspace(0, 2, 5),               # Logarithmic spacing: [1, 3.16, 10, 31.6, 100]
    "kd": [0.1, 0.3, 1.0],                   # 3 damping values
}
# Total: 1 × 19 × 5 × 3 = 285 simulations
```

### **Batch Processing Multiple Robots**
```bash
# Generate datasets for different robot configurations
python3 main.py  # Uses current sweep_config.py

# Edit sweep_config.py for different robot, then:
python3 main.py  # Generates new dataset

# Result: Multiple organized dataset folders
```

---

## Troubleshooting

### **Common Issues**

1. **MuJoCo Installation:**
```bash
pip install mujoco
# If issues, try: pip install mujoco-py
```

2. **Missing XML Files:**
```bash
# Generate XML first
python3 1_xml_generator/1_generate_n_link_robot_xml.py
```

3. **Path Issues:**
```bash
# Always run from 1_data_generation_and_controller/ directory
cd 1_data_generation_and_controller
python3 main.py
```

4. **Visualization Issues:**
```bash
# Disable rendering if display issues
python3 main.py --no-render
```

### **Parameter Validation**
- Joint angles are automatically converted from degrees to radians
- Torque limits are enforced by MuJoCo actuator constraints
- File paths are automatically resolved using relative path detection

---

## Data Format

### **Trajectory JSON Structure**
```json
{
    "metadata": {
        "num_links": 3,
        "num_steps": 2500,
        "dt": 0.002,
        "gravity": [0.0, 0.0, -9.81],
        "solver": "ImplicitFast"
    },
    "static_properties": {
        "nodes": [
            {
                "mass": 0.5,
                "fingertip_mass": 0.031,
                "length": 0.33,
                "radius": 0.025,
                "damping": 0.1,
                "inertia": 0.018225,
                "torque_limit_nm": 100.0
            }
        ],
        "controller_gains": {
            "kp": 10.0,
            "kd": 0.5
        },
        "initial_angle_deg": 45.0,
        "target_angle_deg": -30.0
    },
    "time_series": {
        "theta": [[angle_joint1, angle_joint2, ...]],
        "omega": [[velocity_joint1, velocity_joint2, ...]],
        "alpha": [[acceleration_joint1, acceleration_joint2, ...]],
        "torque": [[torque_joint1, torque_joint2, ...]]
    }
}
```

---

## Authors

**Luca Dofing & Michael Piltz**

This project is part of the Machine Learning for Robotics (AI617) course at KAIST.

---

