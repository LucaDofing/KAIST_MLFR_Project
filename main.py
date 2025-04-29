# main.py

import math, random
import torch
from torch_geometric.loader import DataLoader
from src.datasets import FakePendulumDataset
from src.models import DampingGCN
from src.train import run_training

from src.config import NUM_GRAPHS, TRAIN_SPLIT, BATCH_SIZE
from src.datasets import FakePendulumDataset
from src.models import DampingGCN
from src.train import run_training
import torch
from torch_geometric.loader import DataLoader


# Load dataset
full_dataset = FakePendulumDataset(NUM_GRAPHS) #generates n random graphs with Nodes = joints, Node features = [joint angle, joint velocity], Target labels = true (damping) coefficient per joint
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [TRAIN_SPLIT, NUM_GRAPHS - TRAIN_SPLIT], generator=torch.Generator().manual_seed(42)) #Split into training and testing sets, with a fixed seed
# batches (à 64 graphs) for GNN training.
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# set device (GPU not tested yet)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = DampingGCN().to(device)

# Train model
run_training(model, train_loader, test_loader, device)

# Demo with one random graph, unseen data
print("\nRunning a demo prediction...")
model.eval()
sample = FakePendulumDataset(1)[0].to(device)
with torch.no_grad():
    pred_damping = model(sample)

print("True damping coefficients:", sample.y.squeeze().cpu().numpy())
print("Predicted damping coefficients:", pred_damping.squeeze().cpu().numpy())
