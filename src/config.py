
# src/config.py

# Dataset parameters
NUM_GRAPHS = 5000 # This will be determined by the JSON files now
TOTAL_SAMPLES_FROM_JSON = 4298 # (301-1) + (2000-1) + (2000-1)
TRAIN_SPLIT_RATIO = 0.8 # Use a ratio for splitting
MAX_JOINTS = 1 # Still 1 for these JSONs

# Training parameters
BATCH_SIZE = 64 # Or smaller if memory is an issue with many small graphs
NUM_EPOCHS = 160 # Or more if needed
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

# Model parameters
INPUT_DIM = 2              # (theta, omega)
HIDDEN_DIM = 64
OUTPUT_DIM = 1             # Predict damping per node
NUM_LAYERS = 3             # GCN layers