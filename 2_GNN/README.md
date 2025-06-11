# Graph Neural Network for Robotics Damping Prediction

This project implements a Graph Convolutional Network (GCN) for predicting joint damping coefficients in robotic systems using MuJoCo simulation data. The GNN leverages the graph structure of robotic arms (joints as nodes, links as edges) to learn physical properties from trajectory data.

---

## Project Structure

```
KAIST_MLFR_Project/2_GNN/
│
├── main.py                         # 🎯 Main training and inference pipeline
├── requirements.txt                # Python dependencies
├── Notes                          # Design notes and future improvements
│
├── src/                           # 📦 Core implementation modules
│   ├── __init__.py               
│   ├── config.py                  # Configuration parameters
│   ├── datasets.py                # PyTorch Geometric dataset classes
│   ├── models.py                  # GNN model architectures
│   └── train.py                   # Training and physics simulation functions
│
├── data/                          # 📊 Dataset storage
│   └── mujoco/                    # MuJoCo trajectory data
│       ├── raw/                   # Raw JSON files from simulation
│       └── processed/             # Processed .pt graph data files
│
├── physics_error_propagation.py   # Error analysis and visualization
├── analyze_error_propagation.py   # Interactive error analysis tool
│
└── results/                       # Generated plots and analysis
    └── error_propagation_landscape.png
```

---

## Environment Setup

### **1. Navigate to GNN Directory**
```bash
cd KAIST_MLFR_Project/2_GNN
```

### **2. Create Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch` + `torch-geometric` - PyTorch with graph neural network support
- `numpy`, `scipy` - Numerical computations  
- `matplotlib` - Visualization
- `networkx` - Graph processing utilities

---

## Quick Start Guide

### **Step 1: Prepare Data**
Place your MuJoCo JSON trajectory files in the `data/mujoco/` directory:



**⚠️ Important: Delete Processed Files for New Data**
When adding new JSON files, you MUST delete the processed `.pt` files:
```bash
rm -rf data/mujoco/processed/*.pt
```
This forces the dataset to reprocess all raw JSON files and incorporate the new data.

### **Step 2: Train the GNN**
```bash
python3 main.py
```

This will:
- Load and process JSON trajectory data into graph format
- Split data into training/testing sets (80/20)
- Train a GCN to predict joint damping coefficients
- Save the trained model as `damping_gcn.pth`

### **Step 3: Analyze Results**
```bash
python3 analyze_error_propagation.py
```

---

## Configuration Parameters

Edit `src/config.py` to customize training:

```python
# Dataset parameters
TRAIN_SPLIT_RATIO = 0.8         # 80% training, 20% testing
MAX_JOINTS = 1                  # Currently supports single-link pendulums

# Training parameters  
BATCH_SIZE = 64                 # Batch size for training
NUM_EPOCHS = 50                 # Number of training epochs
LEARNING_RATE = 1e-3            # Learning rate
WEIGHT_DECAY = 1e-5             # L2 regularization

# Model parameters
HIDDEN_DIM = 64                 # Hidden layer dimension
NUM_LAYERS = 3                  # Number of GCN layers
```

---

## Data Format & Processing

### **Input Data (JSON)**
The system expects MuJoCo trajectory JSON files with this structure:
```json
{
    "metadata": {
        "num_links": 1,
        "num_steps": 2500,
        "dt": 0.002,
        "gravity": [0.0, 0.0, -9.81]
    },
    "static_properties": {
        "nodes": [{
            "mass": 0.5,
            "length": 0.33,
            "damping": 0.1,
            "inertia": 0.018225
        }]
    },
    "time_series": {
        "theta": [[angle_t0], [angle_t1], ...],
        "omega": [[velocity_t0], [velocity_t1], ...],
        "alpha": [[acceleration_t0], [acceleration_t1], ...],
        "torque": [[torque_t0], [torque_t1], ...]
    }
}
```

### **Graph Data Processing**
Each timestep transition (t → t+1) becomes a graph:
- **Nodes**: Joints with features `[θ, ω]` (angle, angular velocity)
- **Edges**: Physical connections between joints
- **Target**: Next state `[θ_{t+1}, ω_{t+1}]` for physics prediction
- **Labels**: True damping coefficients for evaluation

### **Processed Files (.pt)**
The dataset automatically converts JSON files to PyTorch Geometric format:
- `mujoco_pendulum_data.pt` - Main processed dataset
- `pre_filter.pt`, `pre_transform.pt` - Processing metadata

**When to Delete .pt Files:**
- Adding new JSON trajectory files
- Changing data processing parameters
- Debugging data loading issues

---

## Model Architecture

### **DampingGCN**
Graph Convolutional Network with:
- **Input**: Node features `[θ, ω, &alpha ]` per joint
- **Graph Convolution**: 3 layers with ReLU activation
- **Output**: Damping coefficient per joint
- **Loss**: Physics-informed MSE on next-state prediction

### **Physics Integration**
The model uses differentiable physics simulation:
```python
def simulate_step_physical(x, applied_torque, estimated_b, dt, mass, length, inertia, gravity):
    theta, omega = x[:, 0], x[:, 1]
    
    # Physics: τ_net = τ_applied + τ_gravity + τ_damping
    torque_gravity = -mass * gravity * length * sin(theta)
    torque_damping = -estimated_b * omega
    net_torque = applied_torque + torque_gravity + torque_damping
    
    # Dynamics: α = τ_net / I
    alpha = net_torque / inertia
    
    # Integration: ω_{t+1} = ω_t + α*dt, θ_{t+1} = θ_t + ω_{t+1}*dt
    omega_next = omega + alpha * dt
    theta_next = theta + omega_next * dt
    
    return [theta_next, omega_next]
```

---

## Error Analysis Tools

### **analyze_error_propagation.py**
Interactive tool for analyzing GNN prediction quality:

```bash
python3 analyze_error_propagation.py
```

**What it does:**
1. Loads a trained GNN model (`damping_gcn.pth`)
2. Selects a sample from the dataset
3. Sweeps through different damping values to create a loss landscape
4. Compares:
   - True damping coefficient performance
   - GNN predicted damping performance  
   - Optimal damping for the physics simulator
5. Generates visualization showing prediction error vs. damping value

**Customization:**
Edit the sample selection in the script:
```python
sample_idx = 150  # Change this to analyze different samples
```

**Output:**
- Console analysis of prediction errors
- Loss landscape plot saved to `results/error_propagation_landscape.png`
- Comparison of true vs. predicted vs. optimal damping values

---


---

## Performance & Scaling

### **Current Capabilities**
- **Single-link pendulums**: Fully supported
- **Multi-link robots**: Framework ready, requires data
- **Dataset size**: Tested with 4,000+ trajectory samples
- **Training time**: ~2-5 minutes for 50 epochs on CPU

### **Memory Usage**
- **Small dataset** (1,000 samples): ~50MB RAM
- **Medium dataset** (5,000 samples): ~200MB RAM
- **Large dataset** (20,000+ samples): Consider batch processing

---

## Future Improvements

As noted in the `Notes` file, potential upgrades include:

### **Advanced Architectures**
- **GAT (Graph Attention Networks)**: Learn which joints matter more dynamically
- **MPNN (Message Passing Neural Networks)**: Model richer interactions between joints  
- **Temporal GNNs**: Capture motion patterns over time sequences

### **Extended Capabilities**
- Multi-link robot support with variable topology
- Real-time parameter adaptation
- Integration with MuJoCo for closed-loop control
- Uncertainty quantification in predictions

---

## Authors

**Luca Dofing & Michael Piltz**

This project is part of the Machine Learning for Robotics (AI617) course at KAIST.

---

## License

This project is part of academic coursework at KAIST. Please respect academic integrity guidelines when using this code.

