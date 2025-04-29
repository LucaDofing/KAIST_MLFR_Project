
# GNN-Based System Identification

This project implements a Graph Neural Network (GNN) pipeline for system identification (SysID) tasks.  
The goal is to learn unknown physical parameters (e.g., joint damping) of simple robotic systems by observing their motion.
---

## Project Structure

```
REPO-NAME/
│
├── src/                  # Source code (datasets, models, training, config)
│   ├── config.py          # Centralized settings (dataset size, model size, etc.)
│   ├── datasets.py        # FakePendulumDataset generator
│   ├── models.py          # GNN architectures (e.g., DampingGCN)
│   ├── train.py           # Training loops and evaluation
│   └── __init__.py
│
├── data/                 # Data folder (currently empty, for future real datasets)
│   ├── fake/             # (synthetic graphs)
│   └── mujoco/           # (real simulation data later)
│
├── experiments/          # Saved training runs, plots, and checkpoints
│   ├── runs/
│   ├── plots/
│   └── checkpoints/
│
├── notebooks/            # Jupyter notebooks for quick experiments
│
├── main.py               # Entry point script
├── requirements.txt      # List of required Python packages
├── README.md             # Project description (this file)
└── .gitignore            # Ignore data/, experiments/, venv, etc.
```

---

## Setup

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/YOUR-TEAM/REPO-NAME.git
cd REPO-NAME

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

*(The project currently runs CPU-only — no CUDA required -- CUDA never tested.)*

---

## Configuration

All important parameters (dataset size, model architecture, training settings)  
are defined in `src/config.py`.

Example:

```python
NUM_GRAPHS = 1200
TRAIN_SPLIT = 1000
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
MAX_JOINTS = 4
```

Modify `config.py` to change the experiment without touching other files.

---

## How to Train

To train a GNN on the synthetic pendulum dataset:

```bash
python main.py
```

This will:
- Generate randomized pendulum graphs
- Train a small GCN model to predict damping from angle/velocity
- Evaluate performance on a held-out test set
- Print a demo prediction for a new unseen graph

Training progress is printed every 20 epochs.

---

## Results

During training, you will see output like:

```
Epoch  20 | Train MSE: 0.0728 | Test MSE: 0.0677
Epoch  40 | Train MSE: 0.0718 | Test MSE: 0.0671
...
```

After training, a sample prediction is shown:

```
True damping coefficients: [0.309, 0.531]
Predicted damping coefficients: [0.419, 0.363]
```

The model learns to predict damping based on the observed state.

---


## Authors

Luca Dofing & Michael Piltz

This project is part of the Machine Learning for Robotics AI617 course at KAIST.

---

