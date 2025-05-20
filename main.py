# main.py

import torch
from torch_geometric.loader import DataLoader
from src.datasets import FakePendulumDataset
from src.models import DampingGCN
from src.train import run_training
from src.config import NUM_GRAPHS, TRAIN_SPLIT, BATCH_SIZE
from src.train import simulate_step


def main():
    # Step 1: Load dataset (unsupervised mode)
    full_dataset = FakePendulumDataset(NUM_GRAPHS, mode="unsupervised")
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [TRAIN_SPLIT, NUM_GRAPHS - TRAIN_SPLIT], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 2: Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DampingGCN().to(device)

    # Step 3: Train
    run_training(model, train_loader, test_loader, device)

    # Optional: Run a test prediction
    sample = full_dataset[0].to(device)
    with torch.no_grad():
        pred_damping = model(sample)
        pred_next = simulate_step(sample.x, pred_damping)

    print("\nSample prediction (unsupervised):")
    print("Current state θ, ω:")
    print(sample.x.cpu().numpy())
    print("Predicted next state:")
    print(pred_next.cpu().numpy())
    print("True next state:")
    print(sample.x_next.cpu().numpy())
    print("Estimated damping per joint (from GNN):")
    print(pred_damping.squeeze().cpu().numpy())

if __name__ == "__main__":
    main()
