# main.py

import math, random
import torch
from torch_geometric.loader import DataLoader
from src.datasets import FakePendulumDataset
from src.models import DampingGCN
from src.train import run_training

# 1. Load dataset
full_dataset = FakePendulumDataset(1200)
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [1000, 200], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Initialize model
model = DampingGCN().to(device)

# 3. Train model
run_training(model, train_loader, test_loader, device)

print("\nRunning a demo prediction...")
model.eval()
sample = FakePendulumDataset(1)[0].to(device)
with torch.no_grad():
    pred_damping = model(sample)

print("True damping coefficients:", sample.y.squeeze().cpu().numpy())
print("Predicted damping coefficients:", pred_damping.squeeze().cpu().numpy())
