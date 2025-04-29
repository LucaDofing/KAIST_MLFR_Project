# src/config.py

# Dataset parameters
NUM_GRAPHS = 1200          # total number of graphs
TRAIN_SPLIT = 1000         # number of graphs for training
MAX_JOINTS = 4             # maximum number of joints in fake pendulums

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Model parameters
INPUT_DIM = 2              # (theta, omega)
HIDDEN_DIM = 64
OUTPUT_DIM = 1             # Predict damping per node
NUM_LAYERS = 3             # GCN layers
